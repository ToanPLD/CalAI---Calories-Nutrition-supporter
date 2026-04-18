import asyncio
from pipelines.ingestion_pipeline import IngestionPipeline
from core.services.qdrant_service import QdrantService


async def main():
    qdrant = QdrantService()
    qdrant.init_collections()

    pipeline = IngestionPipeline(qdrant)
    await pipeline.run()

    print("✅ DONE")


if __name__ == "__main__":
    asyncio.run(main())