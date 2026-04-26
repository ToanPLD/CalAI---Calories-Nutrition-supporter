from core.embedding.clip_service import CLIPService
from core.utils.vector_utils import ensure_list_vector
from core.utils.logger import get_logger

logger = get_logger("ingestion")


class IngestionPipeline:

    def __init__(self, qdrant):
        self.qdrant = qdrant
        self.clip = CLIPService()

    async def run(self, data):

        logger.info("🚀 Starting ingestion...")

        batch = []
        BATCH_SIZE = 64

        for i, item in enumerate(data):

            payload = item.copy()
            payload["domain"] = "food"

            text = f"""
            food: {payload.get('food_name')}
            calories: {payload.get('calories')}
            protein: {payload.get('protein')}
            fat: {payload.get('fat')}
            carb: {payload.get('carb')}
            """

            vector = self.clip.embed_text(text)
            vector = ensure_list_vector(vector)

            if not vector:
                continue

            batch.append({
                "vector": vector,
                "payload": payload
            })

            if len(batch) >= BATCH_SIZE:
                self.qdrant.upsert_batch("food_text_vectors", batch)
                logger.info(f"Inserted {i}")
                batch.clear()

        if batch:
            self.qdrant.upsert_batch("food_text_vectors", batch)

        logger.info("✅ DONE")