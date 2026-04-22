from fastapi import FastAPI, Query
from core.services.query_pipeline import QueryPipeline

from typing import Optional
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService

app = FastAPI()
pipeline = QueryPipeline()

clip = CLIPService()
qdrant = QdrantService()

COLLECTION = "food_text_vectors"


# =========================
# SEARCH API
# =========================
@app.get("/search")
def search(
    query: str,
    min_calories: Optional[float] = None,
    max_calories: Optional[float] = None,
    top_k: int = 10
):
    vec = clip.embed_text(query)

    # ===== FILTER =====
    filters = []
    if min_calories is not None:
        filters.append({"key": "calories", "range": {"gte": min_calories}})
    if max_calories is not None:
        filters.append({"key": "calories", "range": {"lte": max_calories}})

    query_filter = {"must": filters} if filters else None

    results = qdrant.client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True
    )
    @app.get("/query")
    def query(q: str):
        return pipeline.run(q)
    # ===== FORMAT =====
    output = []
    for r in results:
        payload = r.payload
        payload["score"] = r.score
        output.append(payload)

    return output