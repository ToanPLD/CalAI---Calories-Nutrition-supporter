from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from config.settings import settings
from core.embedding.text_embedding_service import TextEmbeddingService
from core.services.retrieval.qdrant_service import QdrantService
from core.services.llm.llm_service import LLMService

router = APIRouter(prefix="/api/qa", tags=["QA"])
_text_embed = None
_qdrant = None
_llm = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=20)
    collections: Optional[List[str]] = None


def get_services():
    global _text_embed, _qdrant, _llm

    if _text_embed is None:
        _text_embed = TextEmbeddingService()
    if _qdrant is None:
        _qdrant = QdrantService()
    if _llm is None:
        _llm = LLMService()

    return _text_embed, _qdrant, _llm


@router.post("/ask")
async def ask_question(req: QueryRequest):

    text_embed, qdrant, llm = get_services()

    # =========================
    # EMBED QUERY
    # =========================
    query_vec = text_embed.embed(req.question)

    collections = req.collections or settings.TEXT_COLLECTIONS
    per_collection = max(1, min(5, req.top_k))

    hits = []
    for collection in collections:
        hits.extend(
            qdrant.search(
                collection_name=collection,
                vector=query_vec,
                top_k=per_collection
            )
        )

    hits.sort(key=lambda h: getattr(h, "score", 0), reverse=True)
    hits = hits[:req.top_k]

    context = [h.payload for h in hits if getattr(h, "payload", None)]

    # =========================
    # ASK LLM
    # =========================
    qa = await llm.answer_question(
        question=req.question,
        context=context
    )

    return {
        "question": req.question,
        "answer": qa["answer"],
        "format": qa.get("format"),
        "context_used": qa.get("context_used", []),
        "context": context
    }
