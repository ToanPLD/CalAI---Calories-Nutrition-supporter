from core.services.vision.qwen_vl_service import QwenVLService
from core.services.clip_service import CLIPService
from core.services.rag.food_rag_service import FoodRAGService
from core.services.rerank.cross_encoder import CrossEncoderReranker
from core.services.nutrition.nutrition_model import NutritionRegressionModel
from core.services.user.user_tracking import UserTrackingService
from core.services.cache.embedding_cache import EmbeddingCache
from core.services.rag.meta_search_pipeline import MetaSearchPipeline

class FoodAnalysisPipeline:

    def __init__(self):
        self.qwen = QwenVLService()
        self.clip = CLIPService()
        self.rag = FoodRAGService()
        self.rerank = CrossEncoderReranker()
        self.nutrition_model = NutritionRegressionModel()
        self.user_tracking = UserTrackingService()
        self.cache = EmbeddingCache()
        self.meta_search = MetaSearchPipeline()

    async def analyze(self, image, user_id=None):

        # ================= STEP 1 =================
        vision = await self.qwen.analyze_food(image)
        query = vision.get("dish_name", "")
        dish = vision.get("dish_name", "")
        desc = vision.get("description", "")
        query_text = dish + " " + desc

        # ================= STEP 2: CACHE =================
        image_vec = self.cache.get_or_set(
            "img_" + dish,
            lambda: self.clip.embed_image(image)
        )

        text_vec = self.cache.get_or_set(
            "txt_" + query_text,
            lambda: self.clip.embed_text(query_text)
        )

        # ================= STEP 3: SEARCH =================
        hits = self.rag.hybrid_search(image_vec, text_vec, dish)

        # ================= STEP 4: RERANK =================
        hits = self.rerank.rerank(query_text, hits)

        best = hits[0].payload if hits else {}

        # ================= STEP 5: NUTRITION =================
        nutrition = self.nutrition_model.predict(image_vec)

        # ================= STEP 6: USER TRACK =================
        

        results = self.meta_search.search(query)
        best = results[0]["data"] if results else {}

        result = {
            "dish_name": dish,
            "confidence": vision.get("confidence", 0),
            "nutrition": nutrition,
            "dish": query,
            "nutrition": best
        }

        if user_id:
            self.user_tracking.log_meal(user_id, result)

        return result
