from qdrant_client import QdrantClient
from config.settings import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

collections = [
    "food_image_vectors",
    "food_text_vectors",
    "beverage_vectors",
    "exercise_vectors",
    "lifestyle_vectors"
]

for col in collections:
    try:
        client.delete_collection(col)
        print(f"🗑 Deleted: {col}")
    except:
        pass

print("✅ All collections removed")