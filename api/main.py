from fastapi import FastAPI
from core.services.query_pipeline import QueryPipeline
from api.routes.food_analysis import router as food_router
from api.routes.qa import router as qa_router

from typing import Optional
from core.embedding.clip_service import CLIPService
from core.services.retrieval.qdrant_service import QdrantService

app = FastAPI()
pipeline = QueryPipeline()
app.include_router(food_router)
app.include_router(qa_router)

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

    results = qdrant.search(
        collection_name=COLLECTION,
        vector=vec,
        top_k=top_k,
        min_calories=min_calories,
        max_calories=max_calories
    )

    # ===== FORMAT =====
    output = []
    for r in results:
        payload = dict(r.payload or {})
        payload["score"] = r.score
        output.append(payload)

    return output


@app.get("/query")
def query(q: str):
    return pipeline.run(q)
