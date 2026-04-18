# import hashlib
# import uuid
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     PointStruct,
#     VectorParams,
#     Distance
# )
# from config.settings import settings


# class QdrantService:


#     def ensure_collection(self, name, dim=512):

#         if not self.client.collection_exists(name):
#             self.client.create_collection(
#                 collection_name=name,
#                 vectors_config=VectorParams(
#                     size=dim,
#                     distance=Distance.COSINE
#             )
#         )

#     def __init__(self):
#         self.client = QdrantClient(
#             url=settings.qdrant_url,
#             api_key=settings.qdrant_api_key
#         )

#         # 🔥 INIT ALL COLLECTIONS
#         self._init_all_collections()

#     # =========================
#     # INIT COLLECTIONS (🔥 FIX 404)
#     # =========================
#     def _init_all_collections(self):

#         collections = {
#             "food_text_vectors": 512,
#             "exercise_vectors": 512,
#             "beverage_vectors": 512,
#             "lifestyle_vectors": 512
#         }

#         for name, dim in collections.items():
#             try:
#                 self.client.get_collection(name)
#                 print(f"✅ Collection exists: {name}")
#             except:
#                 print(f"⚡ Creating collection: {name}")

#                 self.client.create_collection(
#                     collection_name=name,
#                     vectors_config=VectorParams(
#                         size=dim,
#                         distance=Distance.COSINE
#                     )
#                 )

#     # =========================
#     # GENERATE STABLE ID
#     # =========================
#     def _generate_id(self, payload: dict):
#         raw = str(payload)
#         return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

#     # =========================
#     # GENERIC UPSERT
#     # =========================
#     def upsert_generic(self, collection_name, items):

#         points = []

#         for item in items:
#             pid = self._generate_id(item["payload"])

#             points.append(
#                 PointStruct(
#                     id=str(uuid.uuid4()),
#                     vector=item["vector"],
#                     payload=item["payload"]
#                 )
#             )

#         try:
#             self.client.upsert(
#                 collection_name=collection_name,
#                 points=points
#             )

#             print(f"✅ Upserted {len(points)} → {collection_name}")

#         except Exception as e:
#             print(f"❌ Upsert error: {e}")
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config.settings import settings


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

    def ensure_collection(self, name, dim=512):

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {name}")
        else:
            print(f"✅ Collection exists: {name}")

    def upsert_generic(self, collection_name, items):

        points = []

        for item in items:
            points.append(
                PointStruct(
                    id=item["id"],  # 🔥 SHARED ID
                    vector=item["vector"],
                    payload=item["payload"]
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )