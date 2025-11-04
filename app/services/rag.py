import time
import chromadb
import re

from chromadb.utils import embedding_functions
from llama_cpp import Llama
from typing import List, Tuple
from sqlalchemy.future import select

from app.database.models import DocumentChunk
from app.core.config import BASE_DIR, settings
from app.database.session import async_session
from app.services.cache import cache
from app.core.logger import logger
from app.database.crud import get_sources_for_chunks


class RAGService:
    def __init__(self):
        # ---- LLaMA модель ----
        model_path = BASE_DIR / settings.MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        logger.info(f"Загружаем модель LLaMA: {model_path}")

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

        # ---- ChromaDB ----
        self.chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "app/data/chroma"))
        self.collection = self.chroma_client.get_or_create_collection("document_chunks")

        # ---- Встроенный эмбеддер (SentenceTransformer от Chroma) ----
        self.embedder = embedding_functions.DefaultEmbeddingFunction()

        logger.info("RAGService инициализирован (LLaMA + ChromaDB)")

    async def index_chunks(self):
        """
        Загружает все чанки из БД в Chroma (например, при первом запуске).
        """
        async with async_session() as session:
            result = await session.execute(select(DocumentChunk))
            chunks = result.scalars().all()

        if not chunks:
            logger.warning("Нет чанков для индексации.")
            return

        logger.info(f"Добавляем {len(chunks)} чанков в Chroma...")

        ids = [str(c.id) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{"document_id": c.document_id, "chunk_index": c.chunk_index} for c in chunks]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("Индексация завершена")

    async def ask(self, question: str, top_k: int = 5, max_context_chunks: int = 3) -> Tuple[
        str, int, float, List[str]]:
        """
        Вопрос -> Chroma -> LLM ранжировщик -> контекст -> ответ
        """
        start_time = time.time()

        # ---- Проверка кэша ----
        cached_data = await cache.get_cached_answer(question, top_k)
        if cached_data:
            return (
                cached_data["answer"],
                cached_data["tokens"],
                cached_data["duration"],
                cached_data["sources"]
            )

        # ---- 1. Поиск топ чанков в Chroma ----
        query_result = self.collection.query(query_texts=[question], n_results=top_k)
        retrieved_docs = query_result["documents"][0] if query_result["documents"] else []
        chunk_ids = [int(cid) for cid in query_result['ids'][0]]

        if not retrieved_docs:
            return "В базе нет релевантных документов.", 0, 0, []

        # ---- 2. Получаем объекты DocumentChunk и источники ----
        chunks = await get_sources_for_chunks(chunk_ids)
        chunk_map = {c.id: c for c in chunks}

        # ---- 3. Ранжирование через LLM ----
        relevance_scores = []
        for cid, text in zip(chunk_ids, retrieved_docs):
            prompt = f"""
    Оцени, насколько следующий текст отвечает на вопрос.
    Возьми текст и вопрос, и выдай число от 0 до 1, где 1 — текст максимально релевантен.

    Вопрос:
    {question}

    Текст:
    {text}

    Оценка релевантности:"""

            try:
                output = self.llm(
                    prompt,
                    max_tokens=5,
                    temperature=0.0,
                    echo=False
                )
                score_text = output['choices'][0]['text'].strip()
                try:
                    score = float(score_text)
                except ValueError:
                    score = 0.0
            except Exception:
                score = 0.0

            relevance_scores.append((cid, score, text))

        # ---- 4. Берём top N наиболее релевантных ----
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        min_score = 0.7
        top_chunks = [(cid, score, text) for cid, score, text in relevance_scores if score >= min_score]

        if not top_chunks:
            return "В базе нет релевантных документов.", 0, 0, []

        filtered_texts = [text for cid, score, text in top_chunks]
        used_chunk_ids = [cid for cid, score, text in top_chunks]
        sources = list({chunk_map[cid].document.filename for cid in used_chunk_ids})

        # ---- 5. Формируем контекст ----
        context_text = "\n".join(filtered_texts)
        prompt = f"""
        Ты — полезный ассистент. Используй контекст ниже, чтобы ответить на вопрос.
        Если информации недостаточно, скажи об этом.
    
        Контекст:
        {context_text}
    
        Вопрос:
        {question}
    
        Ответ:
        """

        # ---- 6. Генерация ответа ----
        try:
            output = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            answer = output['choices'][0]['text'].strip()
            tokens_used = output.get('usage', {}).get('total_tokens', len(answer) // 4)
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            answer = "Произошла ошибка при генерации ответа."
            tokens_used = 0

        duration = time.time() - start_time

        # ---- 7. Кэширование ----
        cache_data = {
            "answer": answer,
            "tokens": tokens_used,
            "duration": duration,
            "sources": sources
        }
        await cache.set_cached_answer(question, top_k, cache_data)

        return answer, tokens_used, duration, sources


# Глобальный экземпляр
rag = RAGService()

