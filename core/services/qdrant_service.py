# core/services/qdrant_service.py

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance
)
from core.utils.logger import get_logger
from config.settings import settings

logger = get_logger("qdrant")


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60
        )

    # =========================
    # CREATE COLLECTION IF NOT EXISTS
    # =========================
    def ensure_collection(self, name, dim=512):

        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if name in existing:
            print(f"✅ Collection exists: {name}")
            return

        print(f"🆕 Creating collection: {name}")

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

    # =========================
    # ID
    # =========================
    def _generate_id(self, payload):
        raw = str(payload)
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

    # =========================
    # UPSERT
    # =========================
    def upsert_batch(self, collection, batch, retries=3):

        # 🔥 AUTO CREATE COLLECTION
        self.ensure_collection(collection, dim=len(batch[0]["vector"]))

        for attempt in range(retries):
            try:
                points = []

                for item in batch:
                    pid = self._generate_id(item["payload"])

                    points.append(
                        PointStruct(
                            id=pid,
                            vector=item["vector"],
                            payload=item["payload"]
                        )
                    )

                self.client.upsert(
                    collection_name=collection,
                    points=points
                )

                return True

            except Exception as e:
                logger.error(f"❌ Upsert fail (attempt {attempt+1}): {e}")
                import time
                time.sleep(2 * (attempt + 1))

        return False