import asyncio
import io
import os

from PyPDF2 import PdfReader
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from typing import List

from app.database.schema_models import AskResponse, AskRequest
from app.core.logger import logger
from app.services.rag import rag
from app.services.cache import cache
from app.services.other_functions import split_text_into_chunks
from app.core.config import templates
from app.database.crud import save_query, save_document


app = FastAPI(title="Askio")


@app.on_event("startup")
async def startup_event():
    await rag.index_chunks()
    await cache.init_redis()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Загрузка одного или нескольких документов (PDF/TXT/MD).
    Каждый файл автоматически разбивается на чанки и добавляется в базу.
    """
    results = []

    for file in files:
        filename = file.filename
        try:
            logger.info(f"Загружается документ: {filename}")

            # 1. Читаем файл
            content = await file.read()
            if not content:
                raise ValueError("Файл пустой")

            # 2. Определяем расширение и извлекаем текст
            ext = os.path.splitext(filename)[1].lower()

            if ext in [".txt", ".md"]:
                text = content.decode("utf-8", errors="ignore")
            elif ext == ".pdf":
                pdf = PdfReader(io.BytesIO(content))
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Формат {ext} не поддерживается (только .txt, .md, .pdf)",
                )

            # 3. Разбиваем на чанки
            chunks = split_text_into_chunks(text)
            logger.info(f"{filename}: получено {len(chunks)} чанков")

            # 4. Добавляем документ и чанки в БД
            doc_id = await save_document(filename, chunks)

            logger.info(f"{filename}: добавлен в базу (id={doc_id})")

            results.append({
                "filename": filename,
                "status": "ok",
                "document_id": doc_id,
                "chunks": len(chunks),
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Ошибка при обработке {filename}: {e}")
            results.append({
                "filename": filename,
                "status": "error",
                "detail": str(e),
            })

    # Возвращаем суммарный результат
    return {"results": results}


@app.post("/api/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        # Получаем ответ (с кэшированием внутри RAGService)
        answer, tokens_used, duration, sources = await rag.ask(
            request.question,
            request.top_k
        )

        # Асинхронно сохраняем в БД (не блокируем ответ)
        asyncio.create_task(
            save_query(request.question, answer, tokens_used, duration)
        )

        return AskResponse(
            answer=answer,
            tokens=tokens_used,
            latency_ms=duration,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Ошибка в ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
