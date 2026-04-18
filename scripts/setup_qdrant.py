from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config.settings import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

def create(name, dim=512):
    if name in [c.name for c in client.get_collections().collections]:
        print(f"{name} exists")
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
    )
    print(f"Created {name}")


create(settings.qdrant_food_image_collection)
create(settings.qdrant_food_text_collection)
create(settings.qdrant_beverage_collection)
create(settings.qdrant_exercise_collection)
create(settings.qdrant_lifestyle_collection)