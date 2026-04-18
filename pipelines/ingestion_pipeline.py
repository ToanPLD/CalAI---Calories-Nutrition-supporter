from core.services.clip_service import CLIPService
from core.services.embedding_service import EmbeddingService
from data.streaming.hf_stream import HFStreamer


class IngestionPipeline:

    def __init__(self, qdrant):
        self.qdrant = qdrant

        self.clip = CLIPService()
        self.embedder = EmbeddingService()
        self.streamer = HFStreamer()

    async def run(self):

        image_batch = []
        text_batch = []

        async for item in self.streamer.stream():

            # ================= IMAGE =================
            if item.image_path:
                image_vec = self.clip.embed(item.image_path)

                image_batch.append({
                    "vector": image_vec,
                    "payload": item.payload
                })

            # ================= TEXT =================
            if item.text:
                text_vec = self.embedder.embed_text(item.text)

                text_batch.append({
                    "vector": text_vec,
                    "payload": item.payload
                })

            # ================= FLUSH =================
            if len(image_batch) >= 8:
                await self.qdrant.upsert_image_batch(image_batch)
                image_batch.clear()

            if len(text_batch) >= 8:
                await self.qdrant.upsert_text_batch(text_batch)
                text_batch.clear()

        # flush cuối
        if image_batch:
            await self.qdrant.upsert_image_batch(image_batch)

        if text_batch:
            await self.qdrant.upsert_text_batch(text_batch)