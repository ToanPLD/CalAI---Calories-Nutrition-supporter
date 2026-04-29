from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from core.agent.recipe_dataset_agent import RecipeDatasetAgent


router = APIRouter(prefix="/api/recipes", tags=["Recipe Image Dataset"])
_agent = None


class RecipeQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=15)
    intent: Optional[str] = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = RecipeDatasetAgent()
    return _agent


@router.post("/query")
async def query_recipe_dataset(req: RecipeQueryRequest):
    agent = get_agent()
    return await agent.run(
        query=req.question,
        top_k=req.top_k,
        intent=req.intent
    )


@router.get("/search")
def search_recipe_dataset(
    q: str,
    top_k: int = 5,
    intent: Optional[str] = None
):
    agent = get_agent()
    routed_intent, results = agent.retrieve(
        query=q,
        top_k=top_k,
        intent=intent
    )
    return {
        "intent": routed_intent,
        "results": results,
        "citations": [
            result.get("citation")
            for result in results
            if result.get("citation")
        ]
    }
