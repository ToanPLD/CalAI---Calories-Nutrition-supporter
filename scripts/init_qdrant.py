# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance
# import os

# client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY")
# )

# client.recreate_collection(
#     collection_name="food_nutrition_vectors",
#     vectors_config=VectorParams(
#         size=768,
#         distance=Distance.COSINE,
#         on_disk=True
#     )
# )

# print("Qdrant initialized")

# scripts/init_qdrant.py

from qdrant_client import QdrantClient
from config.settings import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

collection_name = settings.qdrant_collection

# ❗ XÓA collection cũ
if client.collection_exists(collection_name):
    print("Deleting old collection...")
    client.delete_collection(collection_name)

# 🔥 TẠO LẠI với dim CLIP = 512
print("Creating new collection...")

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "size": 512,   # 🔥 CLIP DIM
        "distance": "Cosine"
    }
)

print("Done.")