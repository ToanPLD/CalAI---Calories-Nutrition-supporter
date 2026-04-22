from sentence_transformers import CrossEncoder

class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, hits: list):

        pairs = []
        for h in hits:
            text = h.payload.get("dish_name", "") + " " + str(h.payload)
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        reranked = list(zip(hits, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return [h[0] for h in reranked]