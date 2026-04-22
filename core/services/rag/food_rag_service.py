from qdrant_client import QdrantClient, models


class FoodRAGService:

    def __init__(self):
        self.client = QdrantClient("http://localhost:6333")
        self.collection = "food_vectors"

    def hybrid_search(self, image_vec, text_vec, dish_name=None):

        must = []
        if dish_name:
            must.append(
                models.FieldCondition(
                    key="dish_name",
                    match=models.MatchText(text=dish_name)
                )
            )

        query_filter = models.Filter(must=must) if must else None

        image_hits = self.client.search(
            collection_name=self.collection,
            query_vector=image_vec,
            limit=5,
            query_filter=query_filter
        )

        text_hits = self.client.search(
            collection_name=self.collection,
            query_vector=text_vec,
            limit=5
        )

        # merge + rerank nhẹ
        hits = image_hits + text_hits
        hits.sort(key=lambda x: x.score, reverse=True)

        return hits[:5]