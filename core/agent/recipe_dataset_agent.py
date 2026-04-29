import re
import unicodedata

from core.services.llm.llm_service import LLMService
from core.services.rag.recipe_image_rag_service import RecipeImageRAGService


class RecipeDatasetAgent:
    def __init__(self):
        self.rag = RecipeImageRAGService()
        self.llm = LLMService()

    def _normalize(self, text):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def route(self, query):
        q = self._normalize(query)

        if any(keyword in q for keyword in [
            "tim anh", "hinh anh", "anh mon", "image", "photo", "picture"
        ]):
            return "image_retrieval"

        if any(keyword in q for keyword in [
            "so sanh", "compare", "vs", "khac nhau", "nguyen lieu nao"
        ]):
            return "ingredient_comparison"

        if any(keyword in q for keyword in [
            "bien tau", "goi y bien", "thay the", "substitute",
            "multi hop", "ket hop", "extract ingredients"
        ]):
            return "multi_hop"

        if any(keyword in q for keyword in [
            "cong thuc", "recipe", "cach lam", "instructions", "nau"
        ]):
            return "recipe_reasoning"

        return "recipe_reasoning"

    def _extract_ingredient_hint(self, query):
        q = self._normalize(query)
        match = re.search(r"(?:co|voi|with|ingredient)\s+([a-z0-9\-\s]+)", q)
        if not match:
            return None
        return match.group(1).strip()[:80] or None

    def retrieve(self, query, top_k=5, intent=None):
        intent = intent or self.route(query)
        ingredient = self._extract_ingredient_hint(query)

        if intent == "image_retrieval":
            hits = self.rag.search_images(
                query=query,
                top_k=top_k,
                ingredient=ingredient
            )
            return intent, self.rag.format_hits(hits)

        if intent == "multi_hop":
            recipe_hits = self.rag.search_text(
                query=query,
                top_k=max(2, top_k // 2),
                ingredient=ingredient
            )
            image_hits = self.rag.search_images(
                query=query,
                top_k=max(2, top_k - len(recipe_hits)),
                ingredient=ingredient
            )
            merged = list(recipe_hits) + [
                hit for hit in image_hits
                if hit.id not in {recipe_hit.id for recipe_hit in recipe_hits}
            ]
            merged.sort(key=lambda hit: getattr(hit, "score", 0), reverse=True)
            return intent, self.rag.format_hits(merged[:top_k])

        hits = self.rag.search_text(
            query=query,
            top_k=top_k,
            ingredient=ingredient
        )
        return intent, self.rag.format_hits(hits)

    async def run(self, query, top_k=5, intent=None):
        routed_intent, results = self.retrieve(
            query=query,
            top_k=top_k,
            intent=intent
        )

        context = [result["payload"] for result in results]
        qa = await self.llm.answer_question(
            question=query,
            context=context
        )

        return {
            "intent": routed_intent,
            "answer": qa["answer"],
            "format": qa.get("format"),
            "results": results,
            "citations": [
                result.get("citation")
                for result in results
                if result.get("citation")
            ],
            "context_used": qa.get("context_used", [])
        }
