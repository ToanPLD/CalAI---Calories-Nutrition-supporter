from core.services.qdrant_service import QdrantNutritionService
from data.kaggle.ingest_kaggle import KaggleIngest

def main():
    qdrant = QdrantNutritionService()
    ingest = KaggleIngest(qdrant)

    ingest.ingest()


if __name__ == "__main__":
    main()