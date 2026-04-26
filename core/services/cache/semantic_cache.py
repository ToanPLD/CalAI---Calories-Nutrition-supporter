# core/services/cache/semantic_cache.py

import numpy as np
import json
import redis
from config.settings import settings


class SemanticCache:

    def __init__(self):

        print("🧠 Init Semantic Cache...")

        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=int(settings.REDIS_PORT),
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )

        self.prefix = "semantic_cache"
        self.threshold = 0.92 

    def _cosine(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, vector):

        keys = self.client.keys(f"{self.prefix}:*")

        best_score = 0
        best_result = None

        for k in keys:

            data = self.client.get(k)
            if not data:
                continue

            obj = json.loads(data)

            cached_vec = obj["vector"]
            score = self._cosine(vector, cached_vec)

            if score > best_score:
                best_score = score
                best_result = obj["response"]

        if best_score >= self.threshold:
            print(f"⚡ Cache HIT (score={best_score:.3f})")
            return best_result

        print(f"❌ Cache MISS (best={best_score:.3f})")
        return None

    def save(self, vector, query, response):

        key = f"{self.prefix}:{hash(query)}"

        data = {
            "vector": vector,
            "query": query,
            "response": response
        }

        self.client.set(
            key,
            json.dumps(data),
            ex=60 * 60 * 24 
        )

        print("💾 Cached result")