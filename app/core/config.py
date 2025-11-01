from pathlib import Path
from pydantic_settings import BaseSettings
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # корень проекта


class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: int

    class Config:
        env_file = BASE_DIR / ".env.example"  # имя файла с переменными окружения


# Создаём объект settings для использования в проекте
settings = Settings()

TEMPLATES_DIR = BASE_DIR / "app/templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
