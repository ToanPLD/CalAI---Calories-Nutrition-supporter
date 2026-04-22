from core.search.hybrid_search import HybridSearch
from core.search.reranker import Reranker
from core.services.qdrant_service import QdrantService
from core.services.clip_service import CLIPService


class SearchPipeline:

    def __init__(self):
        self.qdrant = QdrantService()
        self.clip = CLIPService()

        self.hybrid = HybridSearch(self.qdrant, self.clip)
        self.reranker = Reranker()
