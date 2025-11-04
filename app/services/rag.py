import time
import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama
from typing import List, Tuple
from sqlalchemy.future import select

from app.database.models import DocumentChunk
from app.core.config import BASE_DIR, settings
from app.database.session import async_session
from app.services.cache import cache
from app.core.logger import logger


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

    async def ask(self, question: str, top_k: int = 5) -> Tuple[str, List[DocumentChunk], int, float]:
        start_time = time.time()

        # ---- Проверка кэша ----
        cached_data = await cache.get_cached_answer(question, top_k)
        if cached_data:
            from app.database.models import DocumentChunk
            chunks = [DocumentChunk(**c) for c in cached_data.get("chunks", [])]
            return (
                cached_data["answer"],
                chunks,
                cached_data["tokens"],
                cached_data["duration"]
            )

        # ---- Поиск релевантных чанков в Chroma ----
        query_result = self.collection.query(query_texts=[question], n_results=top_k)
        retrieved_docs = query_result["documents"][0] if query_result["documents"] else []

        if not retrieved_docs:
            return "В базе нет релевантных документов.", [], 0, 0

        # ---- Формируем контекст ----
        context_text = "\n".join(retrieved_docs)

        prompt = f"""
        Ты — полезный ассистент. Используй контекст ниже, чтобы ответить на вопрос.
        Если информации недостаточно, скажи об этом.
        
        Контекст:
        {context_text}
        
        Вопрос:
        {question}
        
        Ответ:
        """

        # ---- Генерация ответа ----
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

        # ---- Кэширование ----
        cache_data = {
            "answer": answer,
            "chunks": [{"text": t} for t in retrieved_docs],
            "tokens": tokens_used,
            "duration": duration
        }
        await cache.set_cached_answer(question, top_k, cache_data)

        return answer, [], tokens_used, duration


# Глобальный экземпляр
rag = RAGService()

