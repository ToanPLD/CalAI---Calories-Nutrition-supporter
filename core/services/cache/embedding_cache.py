import hashlib

class EmbeddingCache:

    def __init__(self):
        self.cache = {}

    def _hash(self, data: str):
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def get_or_set(self, key, fn):

        h = self._hash(key)

        if h in self.cache:
            return self.cache[h]

        value = fn()
        self.cache[h] = value
        return value