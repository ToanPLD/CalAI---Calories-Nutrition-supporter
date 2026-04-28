import json
import hashlib
from core.services.cache.redis_cache import RedisCache


class EmbeddingCache:

    def __init__(self):
        self.redis = RedisCache()

    def _key(self, text):
        return "embed:" + hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        key = self._key(text)
        data = self.redis.get(key)

        if data:
            return json.loads(data)

        return None

    def set(self, text, vector):
        key = self._key(text)
        self.redis.set(key, json.dumps(vector))

    def get_or_set(self, key, fn):
        val = self.get(key)

        if val is not None:
            return val

    # ⚠️ QUAN TRỌNG: phải gọi fn()
        value = fn()

        if value is None:
            return None

        self.set(key, value)
        return value