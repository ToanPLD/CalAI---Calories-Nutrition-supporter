from core.services.retrieval.qdrant_service import QdrantService
from core.embedding.clip_service import CLIPService
from core.search.bm25_search import BM25Search


class HybridSearch:

    def __init__(self, qdrant_service=None, clip_service=None, collection="food_text_vectors"):
        self.qdrant = qdrant_service or QdrantService()
        self.clip = clip_service or CLIPService()
        self.collection = collection

    def search(self, query, top_k=20):
        vector = self.clip.embed_text(query)
        vector_results = self.qdrant.search(
            collection_name=self.collection,
            vector=vector,
            top_k=top_k
        )

        if not vector_results:
            return []

        texts = [str(doc.payload or {}) for doc in vector_results]
        bm25 = BM25Search(texts)
        bm25_results = bm25.search(query, top_k)

        merged = []
        for idx, bm25_score in bm25_results:
            if idx >= len(vector_results):
                continue

            v_score = vector_results[idx].score
            final_score = 0.6 * v_score + 0.4 * bm25_score
            merged.append((final_score, vector_results[idx]))

        merged.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in merged]

def hybrid_search(query, documents, top_k=20):
    return HybridSearch().search(query, top_k=top_k)
