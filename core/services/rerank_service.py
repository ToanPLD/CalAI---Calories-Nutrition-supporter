import numpy as np


class RerankService:

    @staticmethod
    def rerank(query_vec, results):

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        rescored = []

        for r in results:
            score = cosine(query_vec, r.vector)
            rescored.append((score, r))

        rescored.sort(key=lambda x: x[0], reverse=True)

        return [r[1] for r in rescored]