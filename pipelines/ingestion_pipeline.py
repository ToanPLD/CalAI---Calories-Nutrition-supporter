from core.services.clip_service import CLIPService
from data.streaming.hf_stream import HFStreamer
from core.utils.vector_utils import ensure_list_vector
from core.utils.logger import get_logger

logger = get_logger("ingestion")


class IngestionPipeline:

    def __init__(self, qdrant):
        self.qdrant = qdrant

        # ✅ FIX: import đúng service
        self.clip = CLIPService()
        self.streamer = HFStreamer()

    async def run(self):

        food_image_batch = []
        food_text_batch = []

        logger.info("Starting HF ingestion...")

        async for item in self.streamer.stream():

            payload = item.payload or {}
            payload["domain"] = "food"

            # ================= IMAGE =================
            if getattr(item, "image_path", None):
                image_vec = self.clip.embed_image(item.image_path)
                image_vec = ensure_list_vector(image_vec)

                if image_vec:
                    food_image_batch.append({
                        "vector": image_vec,
                        "payload": payload
                    })

            # ================= TEXT =================
            if getattr(item, "text", None):
                text_vec = self.clip.embed_text(item.text)
                text_vec = ensure_list_vector(text_vec)

                if text_vec:
                    food_text_batch.append({
                        "vector": text_vec,
                        "payload": payload
                    })

            # ================= FLUSH =================
            if len(food_image_batch) >= 8:
                await self.qdrant.upsert_food_image_batch(food_image_batch)
                logger.info(f"Upsert image batch: {len(food_image_batch)}")
                food_image_batch.clear()

            if len(food_text_batch) >= 8:
                await self.qdrant.upsert_food_text_batch(food_text_batch)
                logger.info(f"Upsert text batch: {len(food_text_batch)}")
                food_text_batch.clear()

        # ================= FINAL FLUSH =================
        if food_image_batch:
            await self.qdrant.upsert_food_image_batch(food_image_batch)

        if food_text_batch:
            await self.qdrant.upsert_food_text_batch(food_text_batch)

        logger.info("HF ingestion done")