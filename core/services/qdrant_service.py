import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from config.settings import settings
from core.utils.logger import get_logger

logger = get_logger("qdrant")


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

    def ensure_collection(self, name, dim=512):
        collections = [c.name for c in self.client.get_collections().collections]

        if name in collections:
            return

        logger.info(f"🆕 Creating collection: {name}")

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

    def _generate_id(self, payload: dict):
        raw = str(payload)
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

    def upsert_generic(self, collection_name, items):

        if not items:
            return

        MAX_BATCH = 16 

        for i in range(0, len(items), MAX_BATCH):

            chunk = items[i:i + MAX_BATCH]

            self.ensure_collection(collection_name, dim=len(chunk[0]["vector"]))

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

            try:
                self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            except Exception as e:
                logger.error(f"❌ Upsert error: {e}")
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
            logger.error(f"Search error: {e}")
            return []
        
    def upsert_points(self, collection_name, items):
        try:
            from qdrant_client.models import PointStruct

            points = []

            for idx, item in enumerate(items):
                points.append(
                    PointStruct(
                    id=idx,
                    vector=item["vector"],
                    payload=item["payload"]
                )
            )

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

        except Exception as e:
            logger.error(f"❌ Upsert error: {e}")