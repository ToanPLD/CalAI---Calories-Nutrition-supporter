import uuid
from core.services.qdrant_service import QdrantNutritionService
from core.services.embedding_service import EmbeddingService

from .loaders.fastfood_loader import FastFoodLoader
from .loaders.fruits_loader import FruitsLoader
from .loaders.nutrition_loader import NutritionLoader
from .loaders.daily_loader import DailyLoader

from .processors.schema_mapper import normalize
from .processors.text_builder import build_text


class KaggleIngest:

    def __init__(self, qdrant: QdrantNutritionService):
        self.qdrant = qdrant
        self.embedder = EmbeddingService()

    def ingest(self):

        loaders = [
            FastFoodLoader("data/kaggle/raw/fastfood.csv"),
            FruitsLoader("data/kaggle/raw/fruits.csv"),
            NutritionLoader("data/kaggle/raw/nutrition.csv"),
            DailyLoader("data/kaggle/raw/daily.csv"),
        ]

        for loader in loaders:
            for row in loader.iterate():
                parsed = loader.parse(row)
                normalized = normalize(parsed)

                text = build_text(normalized)
                vector = self.embedder.embed_text(text)

                self.qdrant.upsert({
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": normalized
                })