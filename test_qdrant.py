from core.services.retrieval.qdrant_service import QdrantService

q = QdrantService()

collections = q.client.get_collections()

print(collections)