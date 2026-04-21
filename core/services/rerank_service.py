import numpy as np

class RerankService:

    def rerank(self, query_vec, docs):

        scored = []

        for d in docs:
            vec = np.array(d.vector)
            score = np.dot(query_vec, vec)

            scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [x[0] for x in scored]