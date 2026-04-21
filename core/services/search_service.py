import pandas as pd
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService


class SearchService:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    def search(self, query, collection="food_text_vectors", limit=50):

        vector = self.clip.embed_text(query)

        results = self.qdrant.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit
        )

        data = [r.payload for r in results]

        return pd.DataFrame(data)