# from core.services.clip_service import CLIPService
# from core.services.embedding_service import EmbeddingService
# from core.services.qdrant_service import QdrantService
# from core.services.filter_service import FilterService
# from core.services.rerank_service import RerankService


# class QueryPipeline:

#     def __init__(self):
#         self.clip = CLIPService()
#         self.embedder = EmbeddingService()   # SBERT
#         self.qdrant = QdrantService()
#         self.filter = FilterService()
#         self.rerank = RerankService()

#     def query(self, image_path=None, text=None, filters=None, top_k=5):

#         image_results = []
#         text_results = []

#         # ================= IMAGE QUERY =================
#         if image_path:
#             image_vec = self.clip.embed_image(image_path)

#             image_results = self.qdrant.search_image(image_vec, 30)

#             # 🔥 dùng CLIP text encoder thay vì "food item"
#             text_vec = self.clip.embed_text("food, meal, nutrition")

#             text_results = self.qdrant.search_text(text_vec, 20)

#             query_vec = image_vec  # dùng vector image làm chuẩn

#         # ================= TEXT QUERY =================
#         else:
#             text_vec = self.embedder.embed_text(text)

#             text_results = self.qdrant.search_text(text_vec, 30)

#             # 🔥 map text → image space
#             image_vec = self.clip.embed_text(text)
#             image_results = self.qdrant.search_image(image_vec, 20)

#             query_vec = text_vec

#         # ================= WEIGHT MERGE =================
#         weighted = []

#         for item in image_results:
#             item.score = getattr(item, "score", 1.0) * 1.2  # ưu tiên image
#             weighted.append(item)

#         for item in text_results:
#             item.score = getattr(item, "score", 1.0)
#             weighted.append(item)

#         # ================= REMOVE DUP =================
#         unique = {}
#         for item in weighted:
#             unique[item.id] = item

#         results = list(unique.values())

#         # ================= FILTER =================
#         if filters:
#             results = self.filter.apply(results, filters)

#         # ================= RERANK =================
#         results = self.rerank.rerank(query_vec, results)

#         return results[:top_k]

#     # ================= MULTI-DOMAIN QUERY =================
#     def query_all(self, text, domain=None, filters=None, top_k=5):

#         vec = self.clip.embed_text(text)

#         results = []

#         # 🔥 domain routing
#         if domain == "food":
#             results += self.qdrant.search_food(vec)

#         elif domain == "beverage":
#             results += self.qdrant.search_beverage(vec)

#         elif domain == "exercise":
#             results += self.qdrant.search_exercise(vec)

#         elif domain == "lifestyle":
#             results += self.qdrant.search_lifestyle(vec)

#         else:
#             # search all (fallback)
#             results += self.qdrant.search_food(vec)
#             results += self.qdrant.search_beverage(vec)
#             results += self.qdrant.search_exercise(vec)
#             results += self.qdrant.search_lifestyle(vec)

#         # ================= FILTER =================
#         if filters:
#             results = self.filter.apply(results, filters)

#         # ================= RERANK =================
#         results = self.rerank.rerank(vec, results)

#         return results[:top_k]
    
from core.embedding.clip_service import CLIPService
from core.services.retrieval.qdrant_service import QdrantService
from core.services.rerank_service import RerankService
import re


class QueryPipeline:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()
        self.reranker = RerankService()

    def parse_query(self, query: str):
        """
        simple numeric parser
        """
        min_cal = None
        max_cal = None

        gt_match = re.search(r"(?:calories|calorie|calo|kcal)?\s*>\s*(\d+)", query, re.I)
        lt_match = re.search(r"(?:calories|calorie|calo|kcal)?\s*<\s*(\d+)", query, re.I)

        if gt_match:
            min_cal = int(gt_match.group(1))

        if lt_match:
            max_cal = int(lt_match.group(1))

        cleaned = re.sub(r"(?:calories|calorie|calo|kcal)?\s*[<>]\s*\d+", "", query, flags=re.I)
        return cleaned.strip() or query, min_cal, max_cal

    def run(self, query: str, top_k=10):

        # =========================
        # PARSE
        # =========================
        query_text, min_cal, max_cal = self.parse_query(query)

        # =========================
        # EMBED
        # =========================
        q_vec = self.clip.embed_text(query_text)

        # =========================
        # SEARCH
        # =========================
        results = self.qdrant.search(
            collection_name="exercise_vectors",
            vector=q_vec,
            top_k=top_k,
            min_calories=min_cal,
            max_calories=max_cal,
            with_vectors=True
        )

        # =========================
        # RERANK
        # =========================
        results = self.reranker.rerank(q_vec, results)

        return results

    def query(self, image_path=None, text=None, filters=None, top_k=5):
        if image_path and not text:
            text = "food nutrition"

        if not text:
            return []

        min_cal = None
        max_cal = None
        if filters:
            min_cal = filters.get("min_calories") or filters.get("calories_gt")
            max_cal = filters.get("max_calories") or filters.get("calories_lt")

        query_text, parsed_min, parsed_max = self.parse_query(text)
        min_cal = min_cal if min_cal is not None else parsed_min
        max_cal = max_cal if max_cal is not None else parsed_max

        q_vec = self.clip.embed_text(query_text)
        results = self.qdrant.search(
            collection_name="food_text_vectors",
            vector=q_vec,
            top_k=top_k,
            min_calories=min_cal,
            max_calories=max_cal,
            with_vectors=True
        )

        return self.reranker.rerank(q_vec, results)[:top_k]
