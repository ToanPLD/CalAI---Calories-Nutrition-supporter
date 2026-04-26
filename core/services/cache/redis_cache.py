import redis
from config.settings import settings


class RedisCache:

    def __init__(self):

        print("🔌 Connecting Redis...")
        print("HOST:", settings.REDIS_HOST)
        print("PORT:", settings.REDIS_PORT)

        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5
        )

        # 🔥 TEST CONNECTION NGAY
        try:
            self.client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print("❌ Redis connection FAILED:", e)
            raise e

    def get(self, key):
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key, value, ttl=86400):
        try:
            self.client.setex(key, ttl, value)
        except Exception:
            pass