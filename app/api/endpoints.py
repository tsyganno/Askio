import time
import uuid

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from app.core.config import templates


app = FastAPI(title="Askio")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        txt = (await f.read()).decode(errors="ignore")
        # разбиение на чанки + инжест в vectorstore
        await rag.ingest_document(text=txt, doc_id=str(uuid.uuid4()), metadata={"filename": f.filename})
        saved.append(f.filename)
    return {"uploaded": saved}


@app.post("/api/ask")
async def ask(question: str):
    # кеширование
    cached = await cache.get(question)
    if cached:
        return JSONResponse({"answer": cached["answer"], "sources": cached["sources"], "cached": True})
    start = time.time()
    answer, sources, tokens = await rag.answer(question)
    duration = time.time() - start
    # сохранить в БД (async)
    async with async_session() as s:
        await s.run_sync(lambda sess: None)  # здесь вызов сохранения истории
    # записать в кэш
    await cache.set(question, {"answer": answer, "sources": sources}, ttl=None)
    return {"answer": answer, "sources": sources, "tokens": tokens, "time": duration}
