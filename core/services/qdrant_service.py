import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config.settings import settings
from core.utils.logger import get_logger
from core.utils.retry import retry_async

logger = get_logger("qdrant")


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        self.image_collection = "food_image_vectors"
        self.text_collection = "food_text_vectors"

    # =========================
    # INIT COLLECTIONS
    # =========================
    def init_collections(self):
        """Create collections if not exist"""

        existing = [c.name for c in self.client.get_collections().collections]

        # IMAGE (CLIP → 512)
        if self.image_collection not in existing:
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created image collection")

        # TEXT (MiniLM → 384)
        if self.text_collection not in existing:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created text collection")

    # =========================
    # SEARCH
    # =========================
    def search_image(self, vector, top_k=5):
        try:
            return self.client.search(
                collection_name=self.image_collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []

    def search_text(self, vector, top_k=5):
        try:
            return self.client.search(
                collection_name=self.text_collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Text search error: {e}")
            return []

    # =========================
    # HYBRID SEARCH (IMAGE + TEXT)
    # =========================
    def hybrid_search(self, image_vec, text_vec=None, top_k=5):
        image_results = self.search_image(image_vec, top_k)

        if not text_vec:
            return image_results

        text_results = self.search_text(text_vec, top_k)

        # merge score đơn giản (production có thể dùng rerank)
        merged = {}

        for r in image_results:
            merged[r.id] = {"score": r.score * 0.7, "payload": r.payload}

        for r in text_results:
            if r.id in merged:
                merged[r.id]["score"] += r.score * 0.3
            else:
                merged[r.id] = {"score": r.score * 0.3, "payload": r.payload}

        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # =========================
    # ID
    # =========================
    def _generate_id(self, payload: dict):
        raw = payload.get("food_name", "") + (payload.get("image_path") or "")
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)
    
    def search_image(self, vector, top_k=20):
        return self.client.search(
            collection_name=settings.qdrant_image_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True
    )


    def search_text(self, vector, top_k=20):
        return self.client.search(
            collection_name=settings.qdrant_text_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True
    )


    # =========================
    # UPSERT IMAGE
    # =========================
    @retry_async(max_retries=3)
    async def upsert_image_batch(self, items):
        try:
            points = []

            for item in items:
                pid = self._generate_id(item["payload"])

                points.append(
                    PointStruct(
                        id=pid,
                        vector=item["vector"],  # 512 dim
                        payload=item["payload"]
                    )
                )

            self.client.upsert(
                collection_name=self.image_collection,
                points=points
            )

            logger.info(f"[IMAGE] Upserted {len(points)}")

        except Exception as e:
            logger.error(f"Image upsert error: {e}")

    # =========================
    # UPSERT TEXT
    # =========================
    @retry_async(max_retries=3)
    async def upsert_text_batch(self, items):
        try:
            points = []

            for item in items:
                pid = self._generate_id(item["payload"])

                points.append(
                    PointStruct(
                        id=pid,
                        vector=item["vector"],  # 384 dim
                        payload=item["payload"]
                    )
                )

            self.client.upsert(
                collection_name=self.text_collection,
                points=points
            )

            logger.info(f"[TEXT] Upserted {len(points)}")

        except Exception as e:
            logger.error(f"Text upsert error: {e}")