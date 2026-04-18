# pipelines/query_pipeline.py

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.logger import get_logger

logger = get_logger("query")


class QueryPipeline:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    def run(self, image_path: str):
        vector = self.clip.embed_image(image_path)

        results = self.qdrant.search(vector, top_k=20)

        if not results:
            return {"error": "No results"}

        # 🔥 STEP 1: filter noise
        results = [r for r in results if r.score > 0.2]

        # 🔥 STEP 2: remove duplicates
        seen = set()
        unique = []
        for r in results:
            name = r.payload["food_name"]
            if name not in seen:
                seen.add(name)
                unique.append(r)

        results = unique[:10]

        # 🔥 STEP 3: rerank bằng CLIP text
        results = self.rerank(results)

        # 🔥 STEP 4: weighted aggregation
        total_weight = sum(r.score for r in results)

        calories = sum(r.payload["calories"] * r.score for r in results) / total_weight
        protein = sum(r.payload["protein_g"] * r.score for r in results) / total_weight
        carbs = sum(r.payload["carbs_g"] * r.score for r in results) / total_weight
        fat = sum(r.payload["fat_g"] * r.score for r in results) / total_weight

        # 🔥 STEP 5: confidence score
        confidence = max(r.score for r in results)

        return {
            "nutrition": {
                "calories": round(calories, 2),
                "protein": round(protein, 2),
                "carbs": round(carbs, 2),
                "fat": round(fat, 2),
            },
            "confidence": round(confidence, 3),
            "top_matches": [
                {
                    "food": r.payload["food_name"],
                    "score": round(r.score, 3)
                }
                for r in results[:5]
            ]
        }

    def rerank(self, results):
        reranked = []

        for r in results:
            text_vec = self.clip.embed_text(r.payload["food_name"])

            # cosine similarity (dot vì đã normalize)
            score = sum(a*b for a, b in zip(text_vec, r.vector))

            r.score = (r.score + score) / 2  # combine score
            reranked.append(r)

        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked