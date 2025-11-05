# Этап 1: лёгкий Python-образ
FROM python:3.11-slim AS base

# Устанавливаем системные зависимости (только нужное)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Оптимизация слоёв: сначала зависимости
COPY requirements.txt .

# Устанавливаем зависимости без кеша
RUN pip install --upgrade pip \
    && pip config set global.index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt

# Копируем исходники (код проекта)
COPY . .

# Порт
EXPOSE 8000

# Команда по умолчанию
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]