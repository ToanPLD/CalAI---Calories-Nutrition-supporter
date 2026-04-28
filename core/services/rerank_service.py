import numpy as np

class RerankService:

    def rerank(self, query_vec, docs):
        if not docs:
            return []

        scored = []

        for d in docs:
            vec = getattr(d, "vector", None)
            if vec is None:
                score = getattr(d, "score", 0)
                scored.append((d, score))
                continue

            query_arr = np.array(query_vec)
            doc_arr = np.array(vec)

            if query_arr.shape != doc_arr.shape:
                score = getattr(d, "score", 0)
            else:
                score = float(np.dot(query_arr, doc_arr))

            scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [x[0] for x in scored]
