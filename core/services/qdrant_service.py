import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config.settings import settings


class QdrantService:

    def __init__(self):

        print("👉 QDRANT URL:", settings.QDRANT_URL)

        # 🔥 AUTO detect cloud vs local
        if settings.QDRANT_URL.startswith("https"):
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=False,
                timeout=60.0
            )
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL
            )

    # =========================
    # COLLECTION
    # =========================
    def ensure_collection(self, name, dim=512):

        try:
            collections = [
                c.name for c in self.client.get_collections().collections
            ]

            if name in collections:
                return

            print(f"🆕 Creating collection: {name}")

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )

        except Exception as e:
            print("❌ Qdrant connection error:", e)

    # =========================
    # ID GENERATOR
    # =========================
    def _generate_id(self, payload: dict):
        raw = str(payload)
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

    # =========================
    # UPSERT
    # =========================
    def upsert_generic(self, collection_name, items):

        if not items:
            return

        BATCH_SIZE = 16

        for i in range(0, len(items), BATCH_SIZE):

            chunk = items[i:i + BATCH_SIZE]

            try:
                self.ensure_collection(
                    collection_name,
                    dim=len(chunk[0]["vector"])
                )

                points = []

                for item in chunk:
                    pid = self._generate_id(item["payload"])

                    points.append(
                        PointStruct(
                            id=pid,
                            vector=item["vector"],
                            payload=item["payload"]
                        )
                    )

                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )

            except Exception as e:
                print("❌ Upsert error:", e)

    # =========================
    # SEARCH
    # =========================
    def search(self, collection_name, vector, top_k=5):

        try:
            return self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )

        except Exception as e:
            print("❌ Search error:", e)
            return []

    # =========================
    # HYBRID SEARCH
    # =========================
    def hybrid_search(self, collection, image_vector, text_vector, top_k=5, alpha=0.6):

        try:
            image_hits = self.search(collection, image_vector, top_k)
            text_hits = self.search(collection, text_vector, top_k)

            score_map = {}

            # 🔥 score image
            for hit in image_hits:
                score_map[hit.id] = alpha * hit.score

            # 🔥 score text
            for hit in text_hits:
                if hit.id in score_map:
                    score_map[hit.id] += (1 - alpha) * hit.score
                else:
                    score_map[hit.id] = (1 - alpha) * hit.score

            # 🔥 merge unique
            merged = {hit.id: hit for hit in image_hits + text_hits}

            final_hits = list(merged.values())

            # 🔥 sort
            final_hits.sort(
                key=lambda x: score_map.get(x.id, 0),
                reverse=True
            )

            return final_hits[:top_k]

        except Exception as e:
            print("❌ Hybrid search error:", e)
            return []