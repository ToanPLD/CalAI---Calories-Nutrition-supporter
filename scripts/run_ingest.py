import asyncio
from pipelines.ingestion_pipeline import IngestionPipeline
from core.services.qdrant_service import QdrantService


async def main():
    qdrant = QdrantService()
    pipeline = IngestionPipeline(qdrant)
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())