import json
import redis.asyncio as redis
import hashlib

from typing import Optional

from app.core.config import settings
from app.core.logger import logger


class RedisCache:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

    async def init_redis(self):
        """Инициализация Redis подключения"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis подключен успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Redis: {e}")
            self.redis_client = None
            return False

    def _generate_cache_key(self, question: str, top_k: int) -> str:
        """Генерация ключа кэша на основе вопроса и параметров"""
        content = f"{question}:{top_k}"
        return f"rag_cache:{hashlib.md5(content.encode()).hexdigest()}"

    async def get_cached_answer(self, question: str, top_k: int) -> Optional[dict]:
        """Получить ответ из кэша"""
        if not self.redis_client:
            logger.warning("Redis клиент не инициализирован")
            return None

        try:
            cache_key = self._generate_cache_key(question, top_k)
            logger.info(f"Ищем кэш по ключу: {cache_key}")

            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                logger.info(f"✅ Найден кэш для вопроса: '{question[:30]}...'")
                parsed_data = json.loads(cached_data)
                logger.info(f"Данные кэша: {list(parsed_data.keys())}")
                return parsed_data
            else:
                logger.info(f"❌ Кэш НЕ найден для вопроса: '{question[:30]}...'")

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON из Redis: {e}")
        except Exception as e:
            logger.error(f"Ошибка чтения из Redis: {e}")

        return None

    async def set_cached_answer(self, question: str, top_k: int, data: dict):
        """Сохранить ответ в кэш"""
        if not self.redis_client:
            logger.warning("Redis клиент не инициализирован - пропускаем кэширование")
            return

        try:
            cache_key = self._generate_cache_key(question, top_k)

            cache_data = {
                "question": question,
                "top_k": top_k,
                **data
            }

            logger.info(f"🔄 Сохраняем в кэш ключ: {cache_key}")
            logger.info(
                f"Данные для сохранения: { {k: str(v)[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in cache_data.items()} }")

            result = await self.redis_client.setex(
                cache_key,
                settings.REDIS_CACHE_TTL,
                json.dumps(cache_data, ensure_ascii=False)
            )

            if result:
                logger.info(f"✅ Успешно сохранено в кэш для: '{question[:30]}...'")
            else:
                logger.error("❌ Ошибка: Redis не подтвердил запись")

        except Exception as e:
            logger.error(f"❌ Ошибка записи в Redis: {e}")


# Глобальный объект кэша
cache = RedisCache()
