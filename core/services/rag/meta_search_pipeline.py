from core.services.clip_service import CLIPService
from core.services.qdrant_meta_service import QdrantMetaService


class MetaSearchPipeline:

    def __init__(self):
        self.clip = CLIPService()
        self.meta = QdrantMetaService()

    def search(self, query_text):

        # STEP 1: embed
        vector = self.clip.embed_text(query_text)

        # STEP 2: search meta
        hits = self.meta.search(vector)

        # STEP 3: fetch real data
        results = self.meta.fetch_full_data(hits)

        return results  