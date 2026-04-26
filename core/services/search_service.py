import pandas as pd
from core.embedding.clip_service import CLIPService
from core.services.retrieval.qdrant_service import QdrantService


class SearchService:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    # =========================
    # MAIN SEARCH PIPELINE
    # =========================
    def search(self, query, collection="food_text_vectors", limit=50):

        vector = self.clip.embed_text(query)

        results = self.qdrant.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit
        )

        data = []

        for r in results:
            # ✅ copy để tránh mutate payload gốc
            item = dict(r.payload)

            # ❌ remove image hoàn toàn
            item.pop("image_path", None)

            # score từ vector DB
            item["score"] = r.score

            data.append(item)

        df = pd.DataFrame(data)

        if df.empty:
            return df

        # ✅ DEDUP
        df = self._deduplicate(df)

        # ✅ RERANK semantic nhẹ
        df = self.rerank(df, query)

        return df.reset_index(drop=True)

    # =========================
    # RERANK
    # =========================
    def rerank(self, df, query):

        if df.empty:
            return df

        q = query.lower()

        # boost text match
        df["boost"] = df.apply(
            lambda row: sum(
                1 for v in row.values
                if isinstance(v, str) and q in v.lower()
            ),
            axis=1
        )

        # semantic score nhẹ
        df["nutrition_score"] = (
            df.get("protein", 0) / (df.get("calories", 1) + 1)
        )

        df["final_score"] = (
            df["score"] +
            df["boost"] * 0.2 +
            df["nutrition_score"] * 0.3
        )

        return df.sort_values("final_score", ascending=False)

    # =========================
    # DEDUP
    # =========================
    def _deduplicate(self, df: pd.DataFrame):

        if df.empty:
            return df

        # ưu tiên id
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])

        else:
            subset_cols = [
                col for col in ["food_name", "calories", "protein", "carb", "fat"]
                if col in df.columns
            ]

            if subset_cols:
                df = df.drop_duplicates(subset=subset_cols)

        return df.reset_index(drop=True)