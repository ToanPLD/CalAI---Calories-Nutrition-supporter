from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, results):

        pairs = []

        for r in results:
            text = " ".join([str(v) for v in r.payload.values()])
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in reranked[:5]]