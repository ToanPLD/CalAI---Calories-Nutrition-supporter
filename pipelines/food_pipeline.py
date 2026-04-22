from core.services.vision.qwen_vl_service import QwenVLService
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.services.llm_service import LLMService


class FoodPipeline:

    def __init__(self):
        self.vision = QwenVLService()
        self.clip = CLIPService()
        self.qdrant = QdrantService()
        self.llm = LLMService()

    async def run(self, image):

        # =========================
        # STEP 2: VISION MODEL
        # =========================
        vision_result = await self.vision.analyze_food(image)

        dish_name = vision_result.get("dish_name", "")
        description = vision_result.get("description", "")

        # =========================
        # STEP 3: EMBEDDING
        # =========================
        image_vec = self.clip.embed_image_pil(image)
        text_vec = self.clip.embed_text(dish_name + " " + description)

        # =========================
        # STEP 3: HYBRID SEARCH (MULTI COLLECTION)
        # =========================
        collections = [
            "food_text_vectors",
            "beverage_vectors",
            "exercise_vectors"
        ]

        all_hits = []

        for col in collections:
            hits = self.qdrant.hybrid_search(
                collection=col,
                image_vector=image_vec,
                text_vector=text_vec,
                top_k=3,
                alpha=0.6
            )
            all_hits.extend(hits)

        # 🔥 sort SAU loop (rất quan trọng)
        all_hits.sort(key=lambda x: x.score, reverse=True)

        rag_results = all_hits[:5]

        # =========================
        # EXTRACT NUTRITION
        # =========================
        nutrition = self._extract_nutrition(rag_results)

        # =========================
        # STEP 4: GENERATE FINAL RESPONSE
        # =========================
        final = await self.llm.generate_final(
            vision=vision_result,
            nutrition=nutrition
        )

        return final

    # =========================
    # HELPER
    # =========================
    def _extract_nutrition(self, results):

        if not results:
            return {}

        best = results[0].payload

        return {
            "calories": best.get("calories"),
            "protein": best.get("protein"),
            "carbs": best.get("carbs"),
            "fat": best.get("fat")
        }