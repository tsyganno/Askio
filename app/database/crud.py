from app.database.session import async_session
from app.database.models import QueryHistory, Document, DocumentChunk
from app.core.config import created_at_irkutsk_tz
from app.core.logger import logger


async def save_document(filename: str, chunks: list):
    """Сохранение документа в базу данных"""
    try:
        async with async_session() as session:
            doc = Document(filename=filename, chunks_count=len(chunks))
            session.add(doc)
            await session.flush()  # получаем ID документа

            # Добавляем чанки
            for i, text in enumerate(chunks):
                chunk = DocumentChunk(
                    document_id=doc.id,
                    text=text,
                    chunk_index=i
                )
                session.add(chunk)

            await session.commit()
            logger.info(f"Документ сохранен в БД: {filename}...")
            return doc.id
    except Exception as e:
        logger.error(f"Ошибка сохранения документа в БД: {e}")


async def save_query(question: str, answer: str, tokens: int, latency_ms: float):
    """Сохранение запроса в базу данных"""
    try:
        async with async_session() as session:
            query = QueryHistory(
                question=question,
                answer=answer,
                tokens=tokens,
                latency_ms=latency_ms,
                created_at=created_at_irkutsk_tz
            )
            session.add(query)
            await session.commit()
            logger.info(f"Запрос сохранен в БД: {question[:50]}...")
    except Exception as e:
        logger.error(f"Ошибка сохранения запроса в БД: {e}")
