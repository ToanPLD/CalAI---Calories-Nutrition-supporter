from core.services.qdrant_service import QdrantService

q = QdrantService()

collections = q.client.get_collections()

print(collections)