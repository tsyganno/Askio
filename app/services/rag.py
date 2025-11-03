import time

from llama_cpp import Llama
from sqlalchemy.future import select
from typing import List, Tuple


from app.database.models import DocumentChunk
from app.core.config import BASE_DIR, settings
from app.database.session import async_session
from app.services.cache import cache
from app.core.logger import logger


class RAGService:
    def __init__(self):
        # Путь к скачанной GGUF модели
        model_path = BASE_DIR / settings.MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

        logger.info(f"Загружаем модель: {model_path}")

        # Инициализация модели
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Размер контекста
            n_threads=4,  # Количество потоков
            verbose=False  # Отключить лишние логи
        )

        logger.info("Модель успешно загружена")

    async def get_all_chunks(self) -> List[DocumentChunk]:
        async with async_session() as session:
            result = await session.execute(select(DocumentChunk))
            return result.scalars().all()

    def _count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте (упрощенный)"""
        # Для точного подсчета нужно использовать токенайзер модели
        # Это упрощенная версия - примерно 1 токен = 4 символа
        return len(text) // 4

    async def ask(self, question: str, top_k: int = 5) -> Tuple[str, List[DocumentChunk], int, float]:
        start_time = time.time()

        # Проверяем кэш
        cached_data = await cache.get_cached_answer(question, top_k)
        if cached_data:
            # Восстанавливаем объекты чанков из данных
            from app.database.models import DocumentChunk
            chunks = [DocumentChunk(**chunk_data) for chunk_data in cached_data.get("chunks", [])]

            # ВОЗВРАЩАЕМ ДАННЫЕ ИЗ КЭША (включая вопрос)
            return (
                cached_data["answer"],
                chunks,
                cached_data["tokens"],
                cached_data["duration"]
            )

        chunks = await self.get_all_chunks()
        if not chunks:
            return "В базе нет документов.", [], 0, 0

        # Поиск релевантных чанков
        question_words = set(question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            score = len(question_words & chunk_words)
            if score > 0:
                scored_chunks.append((score, chunk))

        # Сортировка и выбор топ-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [c for _, c in scored_chunks[:top_k]]

        context_text = "\n".join([c.text for c in relevant_chunks]) or "Нет релевантных документов."

        # Формируем промпт для LLaMA 3.2
        prompt = f"""<|start_header_id|>system<|end_header_id|>

        Ты - полезный AI ассистент. Ответь на вопрос пользователя используя предоставленный контекст.
        Если в контексте нет информации для ответа, скажи об этом.
        
        Контекст: {context_text}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
        {question}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        """

        # Генерация ответа
        generation_start = time.time()
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
            tokens_used = output['usage']['total_tokens'] if 'usage' in output else self._count_tokens(answer)

        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            answer = "Извините, произошла ошибка при генерации ответа."
            tokens_used = 0

        duration = time.time() - start_time
        generation_duration = time.time() - generation_start

        # Сохраняем в кэш (ВКЛЮЧАЯ ВОПРОС)
        cache_data = {
            "answer": answer,
            "chunks": [{"id": c.id, "text": c.text} for c in relevant_chunks],
            "tokens": tokens_used,
            "duration": generation_duration
            # Вопрос сохраняется автоматически в redis_service.set_cached_answer
        }
        await cache.set_cached_answer(question, top_k, cache_data)

        return answer, relevant_chunks, tokens_used, duration


# Глобальный объект RAG
rag = RAGService()
