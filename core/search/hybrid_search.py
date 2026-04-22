from core.services.qdrant_service import QdrantService
from core.services.clip_service import CLIPService
from core.search.bm25_search import BM25Search

qdrant = QdrantService()
clip = CLIPService()


def hybrid_search(query, documents, top_k=20):

    # ===== VECTOR SEARCH =====
    vector = clip.embed_text(query)

    vector_results = qdrant.client.search(
        collection_name="food_text_vectors",
        query_vector=vector,
        limit=top_k
    )

    # ===== BM25 =====
    texts = [str(doc.payload) for doc in vector_results]
    bm25 = BM25Search(texts)
    bm25_results = bm25.search(query, top_k)

    # ===== MERGE SCORE =====
    merged = []

    for i, (idx, bm25_score) in enumerate(bm25_results):
        v_score = vector_results[idx].score

        final_score = 0.6 * v_score + 0.4 * bm25_score

        merged.append((final_score, vector_results[idx]))

    merged.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in merged]