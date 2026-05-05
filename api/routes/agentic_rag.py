from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from core.agent.agentic_rag import AgenticRAG


router = APIRouter(prefix="/api/agent", tags=["Agentic RAG"])
_agentic_rag = None


class AgenticQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=6, ge=1, le=20)
    intent: Optional[str] = None
    session_id: Optional[str] = None
    conversation_context: Optional[str] = None
    is_follow_up: Optional[bool] = None
    user_profile: Optional[Dict[str, Any]] = None


def get_agentic_rag():
    global _agentic_rag
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG()
    return _agentic_rag


@router.post("/query")
async def query_agentic_rag(req: AgenticQueryRequest):
    agent = get_agentic_rag()
    return await agent.run(
        query=req.question,
        top_k=req.top_k,
        intent=req.intent,
        session_id=req.session_id,
        conversation_context=req.conversation_context,
        is_follow_up=req.is_follow_up,
        user_profile=req.user_profile
    )


@router.get("/query")
async def query_agentic_rag_get(
    q: Optional[str] = None,
    question: Optional[str] = Query(default=None),
    top_k: int = 6,
    intent: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_context: Optional[str] = None,
    is_follow_up: Optional[bool] = None
):
    final_query = q or question
    if not final_query:
        raise HTTPException(status_code=422, detail="Missing query parameter `q` or `question`.")

    agent = get_agentic_rag()
    return await agent.run(
        query=final_query,
        top_k=top_k,
        intent=intent,
        session_id=session_id,
        conversation_context=conversation_context,
        is_follow_up=is_follow_up
    )
