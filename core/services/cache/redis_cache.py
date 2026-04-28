import redis
import json
from config.settings import settings


class RedisCache:

    def __init__(self):

        print("🔌 Connecting Redis...")
        print("HOST:", settings.REDIS_HOST)
        print("PORT:", settings.REDIS_PORT)

        self.client = None
        self.memory = {}

        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,
                socket_connect_timeout=5
            )
            self.client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print("⚠️ Redis connection FAILED, using in-memory cache:", e)
            self.client = None

    def _encode(self, value):
        if isinstance(value, (bytes, str)):
            return value
        return json.dumps(value)

    def get(self, key):
        try:
            if self.client is None:
                return self.memory.get(key)
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key, value, ttl=86400):
        try:
            value = self._encode(value)
            if self.client is None:
                self.memory[key] = value
                return
            self.client.setex(key, ttl, value)
        except Exception:
            pass

    def ttl(self, key):
        try:
            if self.client is None:
                return None
            return self.client.ttl(key)
        except Exception:
            return None
