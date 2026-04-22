from qdrant_client import QdrantClient, models


class QdrantMetaService:

    def __init__(self):
        self.client = QdrantClient("http://localhost:6333")

        self.meta_collection = "meta_vectors"

        self.collections = {
            "food": "food_vectors",
            "beverage": "beverage_vectors",
            "exercise": "exercise_vectors",
            "lifestyle": "lifestyle_vectors",
            "diet": "diet_recommendations_vectors",
            "recipe": "food_recipes_vectors",
            "recipe": "recipes_vectors"
        }

        self._init_collections()

    # ================= INIT =================

    def _init_collections(self):

        if self.meta_collection not in [
            c.name for c in self.client.get_collections().collections
        ]:
            self.client.create_collection(
                collection_name=self.meta_collection,
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            )

    # ================= UPSERT META =================

    def upsert_meta(self, items):

        """
        items:
        [
            {
                "id": int,
                "vector": [...],
                "payload": {
                    "name": "...",
                    "domain": "food",
                    "ref_id": 123
                }
            }
        ]
        """

        self.client.upsert(
            collection_name=self.meta_collection,
            points=[
                models.PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"]
                )
                for item in items
            ]
        )

    # ================= SEARCH =================

    def search(self, vector, top_k=5):

        hits = self.client.search(
            collection_name=self.meta_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )

        return hits

    # ================= FETCH FULL DATA =================

    def fetch_full_data(self, hits):

        results = []

        for h in hits:
            payload = h.payload

            domain = payload["domain"]
            ref_id = payload["ref_id"]

            collection = self.collections[domain]

            point = self.client.retrieve(
                collection_name=collection,
                ids=[ref_id],
                with_payload=True
            )

            if point:
                results.append({
                    "meta_score": h.score,
                    "data": point[0].payload
                })

        return results