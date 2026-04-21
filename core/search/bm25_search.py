from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, documents):
        self.docs = documents
        self.tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, top_k=20):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]