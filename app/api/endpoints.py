from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from app.core.config import templates


app = FastAPI(title="Askio")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
