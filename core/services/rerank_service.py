import numpy as np


class RerankService:

    def rerank(self, query_vec, items):
        scored = []

        q = np.array(query_vec)

        for item in items:
            v = np.array(item.vector)

            # cosine similarity
            score = np.dot(q, v)

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [x[1] for x in scored]