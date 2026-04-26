from core.services.vision.qwen_vl_service import QwenVLService
from core.embedding.clip_service import CLIPService
from core.services.retrieval.qdrant_service import QdrantService
from core.services.llm.llm_service import LLMService
from core.services.cache.semantic_cache import SemanticCache
from core.embedding.text_embedding_service import TextEmbeddingService


class FoodPipeline:

    def __init__(self):
        self.vision = QwenVLService()
        self.clip = CLIPService()
        self.text_embed = TextEmbeddingService()

        self.qdrant = QdrantService()
        self.llm = LLMService()
        self.semantic_cache = SemanticCache()

        self.text_collections = list(set([
            "beverage_text_vectors_768",
            "exercise_text_vectors_768",
            "food_text_vectors_768",
            "diet_recommendations_vectors",
            "exercise_vectors_768",
            "food_vectors_768",
            "food_nutrition_vectors_768",
            "food_nutrition_dev_vectors_768",
            "food_fruit_vectors_768",
            "food_global_10k_vectors_768",
            "exercise_gym_vectors_768",
            "lifestyle_vectors_768",
            "food_common_vectors_768",
            "lifestyle_obesity_vectors_768",
            "recipes_vectors_768",
            "beverage_vectors_768",
            "food_recipes_vectors_768"
        ]))

    async def run(self, image):

        try:

            image_vec = self.clip.embed_image_pil(image)

            if image_vec is None:
                return {"error": "Embedding failed"}

            vision_result = await self.vision.analyze_food(image)

            dish_name = vision_result.get("dish_name", "")
            description = vision_result.get("description", "")

            text_vec = self.text_embed.embed(
                f"{dish_name}. {description}"
            )

            query_vec = self._combine_vec(image_vec, text_vec)

            cached = self.semantic_cache.search(query_vec)

            if cached:
                print("⚡ Semantic cache HIT")
                return cached

            rag_results = self._multi_collection_search(text_vec)

            nutrition = self._extract_nutrition(rag_results)

            final = await self.llm.generate_final(
                vision=vision_result,
                nutrition=nutrition,
                rag=rag_results
            )

            self.semantic_cache.save(
                vector=query_vec,
                query=dish_name,
                response=final
            )

            return final

        except Exception as e:
            print("❌ Pipeline error:", e)
            return {
                "error": str(e),
                "message": "Pipeline failed"
            }

    def _combine_vec(self, img, txt, w=0.6):
        return [(w * i + (1 - w) * t) for i, t in zip(img, txt)]

    def _multi_collection_search(self, text_vec):

        all_hits = []

        for col in self.text_collections:

            hits = self.qdrant.search(
                collection_name=col,
                vector=text_vec,
                top_k=3
            )

            if hits:
                all_hits.extend(hits)

        all_hits.sort(
            key=lambda x: getattr(x, "score", 0),
            reverse=True
        )

        return all_hits[:5]

    def _extract_nutrition(self, results):

        if not results:
            return {
                "calories": None,
                "protein": None,
                "carbs": None,
                "fat": None
            }

        best = results[0].payload or {}

        return {
            "calories": best.get("calories"),
            "protein": best.get("protein"),
            "carbs": best.get("carbs"),
            "fat": best.get("fat")
        }