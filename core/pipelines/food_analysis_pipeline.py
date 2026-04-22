from core.services.vision.qwen_vl_service import QwenVLService
from core.services.clip_service import CLIPService
from core.services.rag.food_rag_service import FoodRAGService
from core.services.nutrition_calculator import NutritionCalculator


class FoodAnalysisPipeline:

    def __init__(self):
        self.qwen = QwenVLService()
        self.clip = CLIPService()
        self.rag = FoodRAGService()
        self.nutrition = NutritionCalculator()

    async def analyze(self, image):

        # =========================
        # STEP 1: VISION
        # =========================
        vision = await self.qwen.analyze_food(image)

        dish = vision.get("dish_name", "")
        desc = vision.get("description", "")

        # =========================
        # STEP 2: EMBEDDING
        # =========================
        image_vec = self.clip.embed_image(image)
        text_vec = self.clip.embed_text(dish + " " + desc)

        # =========================
        # STEP 3: RAG
        # =========================
        hits = self.rag.hybrid_search(image_vec, text_vec, dish)

        payload = hits[0].payload if hits else {}

        # =========================
        # STEP 4: NUTRITION
        # =========================
        nutrition = self.nutrition.calculate_simple(payload)

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "dish_name": dish,
            "confidence": vision.get("confidence", 0),
            "ingredients": vision.get("visible_ingredients", []),
            "nutrition": nutrition,
            "rag_match": payload.get("dish_name")
        }