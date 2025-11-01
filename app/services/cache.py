import aioredis
import os


class Cache:
    def __init__(self):
        self.url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.client = None

    async def init(self):
        self.client = await aioredis.from_url(self.url)

    async def get(self, key):
        raw = await self.client.get(key)
        if raw:
            import json
            return json.loads(raw)
        return None

    async def set(self, key, value, ttl=None):
        import json
        await self.client.set(key, json.dumps(value), ex=ttl or int(os.environ.get("CACHE_TTL_SECONDS", 3600)))


cache = Cache()

