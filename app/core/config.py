import pytz

from datetime import datetime
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # корень проекта


class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: int
    DATABASE_URL: str
    REDIS_URL: str
    REDIS_CACHE_TTL: int
    MODEL_PATH: str

    class Config:
        env_file = BASE_DIR / ".env"  # имя файла с переменными окружения


TEMPLATES_DIR = BASE_DIR / "app/templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

irkutsk_tz = pytz.timezone("Asia/Irkutsk")
created_at_irkutsk_tz = datetime.now(irkutsk_tz)

# Создаём объект settings для использования в проекте
settings = Settings()
