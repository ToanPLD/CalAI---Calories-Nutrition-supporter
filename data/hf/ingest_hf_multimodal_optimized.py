# data/hf/ingest_hf_multimodal_optimized.py

import asyncio
from datasets import load_dataset
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row

TEXT_COLLECTION = "hf_food_text"

BATCH_SIZE = 64

clip = CLIPService()
qdrant = QdrantService()


async def main():
    print("🔥 ENTER MAIN")

    print("⏳ Loading dataset...")
    dataset = load_dataset("pinkieseb/nutrition_dataset", split="train")
    print("✅ Dataset loaded:", len(dataset))

    batch = []

    for i, item in enumerate(dataset):

        # =========================
        # BUILD PAYLOAD
        # =========================
        payload = dict(item)

        # ❌ REMOVE PIL IMAGE OBJECT
        if "image" in payload:
            payload.pop("image")

        # ✅ ADD IMAGE URL / PATH (nếu có)
        if "image_path" in item:
            payload["image_url"] = item["image_path"]

        # =========================
        # TEXT
        # =========================
        text = serialize_row(payload)

        if not text:
            continue

        # =========================
        # EMBEDDING
        # =========================
        vector = clip.embed_text(text)

        if vector is None:
            continue

        batch.append({
            "vector": vector.tolist(),
            "payload": payload
        })

        # =========================
        # UPSERT
        # =========================
        if len(batch) >= BATCH_SIZE:
            qdrant.upsert_batch(TEXT_COLLECTION, batch)
            print(f"📝 Inserted {i}")
            batch.clear()

        # =========================
        # DEBUG PROGRESS
        # =========================
        if i % 1000 == 0:
            print(f"📦 Processed {i}")

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_batch(TEXT_COLLECTION, batch)

    print("✅ DONE")


if __name__ == "__main__":
    asyncio.run(main())