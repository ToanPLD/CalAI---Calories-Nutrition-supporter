from core.embedding.text_embedding_service import TextEmbeddingService
from core.services.qdrant_meta_service import QdrantMetaService


class MetaSearchPipeline:

    def __init__(self):
        self.text_embed = TextEmbeddingService()
        self.meta = QdrantMetaService()

    def search(self, query_text):

        # STEP 1: embed
        vector = self.text_embed.embed(query_text)

        # STEP 2: search meta
        hits = self.meta.search(vector)

        # STEP 3: fetch real data
        results = self.meta.fetch_full_data(hits)

        return results  
