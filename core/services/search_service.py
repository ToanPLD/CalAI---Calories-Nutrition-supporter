import pandas as pd
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService


class SearchService:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    def search(self, query, collection="food_text_vectors", limit=50):

        # ================= EMBED QUERY =================
        vector = self.clip.embed_text(query)

        if vector is None:
            return pd.DataFrame()

        # ================= SEARCH =================
        results = self.qdrant.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True
        )

        # ================= EXTRACT =================
        data = [r.payload for r in results if r.payload]

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # ================= 🔥 DEDUPLICATE =================
        df = self._deduplicate(df)

        return df

    # =========================
    # DEDUP LOGIC
    # =========================
    def _deduplicate(self, df: pd.DataFrame):

        # 👉 nếu có id thì ưu tiên id
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])

        # 👉 fallback: dùng numeric fields
        else:
            subset_cols = []

            for col in ["food_name", "calories", "protein", "carb", "fat"]:
                if col in df.columns:
                    subset_cols.append(col)

            if subset_cols:
                df = df.drop_duplicates(subset=subset_cols)

        # reset index cho sạch
        df = df.reset_index(drop=True)

        return df