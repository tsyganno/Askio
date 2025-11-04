import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_ask_no_relevant_documents():
    """Если Chroma не вернула документы, RAG возвращает сообщение об отсутствии релевантных"""
    with patch("app.services.rag.Llama") as mock_llama, \
         patch("app.services.rag.chromadb.PersistentClient") as mock_chroma, \
         patch("app.services.rag.BASE_DIR", Path("/fake/path")), \
         patch("app.services.rag.settings") as mock_settings, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("app.services.rag.cache") as mock_cache:

        mock_settings.MODEL_PATH = "fake_model.gguf"
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {"documents": [[]], "ids": [[]]}

        # ---- Важно: используем AsyncMock для асинхронных методов ----
        mock_cache.get_cached_answer = AsyncMock(return_value=None)
        mock_cache.set_cached_answer = AsyncMock()

        from app.services.rag import RAGService
        service = RAGService()
        service.llm = mock_llama.return_value
        service.collection = mock_collection

        answer, tokens, duration, sources = await service.ask("тестовый вопрос")

        assert answer == "В базе нет релевантных документов."
        assert tokens == 0
        assert duration == 0
        assert sources == []


@pytest.mark.asyncio
async def test_ask_uses_cache():
    """Если есть кэш, Chroma и LLM не вызываются"""
    with patch("app.services.rag.Llama") as mock_llama, \
         patch("app.services.rag.chromadb.PersistentClient") as mock_chroma, \
         patch("app.services.rag.BASE_DIR", Path("/fake/path")), \
         patch("app.services.rag.settings") as mock_settings, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("app.services.rag.cache") as mock_cache:

        mock_settings.MODEL_PATH = "fake_model.gguf"
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        cached_data = {
            "answer": "Кэшированный ответ",
            "tokens": 12,
            "duration": 0.1,
            "sources": ["cached.pdf"]
        }
        mock_cache.get_cached_answer = AsyncMock(return_value=cached_data)
        mock_cache.set_cached_answer = AsyncMock()

        from app.services.rag import RAGService
        service = RAGService()
        service.llm = mock_llama.return_value
        service.collection = mock_collection

        answer, tokens, duration, sources = await service.ask("вопрос в кэше")

        assert answer == "Кэшированный ответ"
        assert tokens == 12
        assert duration == 0.1
        assert sources == ["cached.pdf"]
        mock_collection.query.assert_not_called()
        mock_llama.return_value.assert_not_called()


@pytest.mark.asyncio
async def test_ask_with_relevant_document():
    """Если есть релевантные документы, ответ формируется корректно"""
    with patch("app.services.rag.Llama") as mock_llama, \
         patch("app.services.rag.chromadb.PersistentClient") as mock_chroma, \
         patch("app.services.rag.BASE_DIR", Path("/fake/path")), \
         patch("app.services.rag.settings") as mock_settings, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("app.services.rag.cache") as mock_cache, \
         patch("app.services.rag.get_sources_for_chunks", new_callable=AsyncMock) as mock_get_sources:

        mock_settings.MODEL_PATH = "fake_model.gguf"
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "documents": [["Текст документа"]],
            "ids": [["1"]]
        }

        mock_chunk = Mock()
        mock_chunk.id = 1
        mock_chunk.document.filename = "doc1.pdf"
        mock_get_sources.return_value = [mock_chunk]

        # Мокаем LLM: сначала оценка релевантности, потом генерация ответа
        mock_llm_instance = Mock()
        mock_llm_instance.side_effect = [
            {'choices': [{'text': '0.9'}], 'usage': {'total_tokens': 1}},
            {'choices': [{'text': 'Ответ на вопрос'}], 'usage': {'total_tokens': 10}}
        ]
        mock_llama.return_value = mock_llm_instance

        mock_cache.get_cached_answer = AsyncMock(return_value=None)
        mock_cache.set_cached_answer = AsyncMock()

        from app.services.rag import RAGService
        service = RAGService()
        service.llm = mock_llm_instance
        service.collection = mock_collection

        answer, tokens, duration, sources = await service.ask("тестовый вопрос")

        assert answer == "Ответ на вопрос"
        assert tokens == 10
        assert isinstance(duration, float)
        assert sources == ["doc1.pdf"]
