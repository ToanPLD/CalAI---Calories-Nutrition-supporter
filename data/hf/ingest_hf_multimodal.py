import os
import uuid
from datasets import load_dataset
from PIL import Image

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService


# =========================
# CONFIG
# =========================
TEXT_COLLECTION = "food_text_vectors"
IMAGE_COLLECTION = "food_image_vectors"
IMAGE_DIR = "data/storage/images"

os.makedirs(IMAGE_DIR, exist_ok=True)


# =========================
# SERIALIZE TEXT (FIX CLIP 77 TOKENS)
# =========================
def serialize_row(payload: dict, max_chars=300):
    parts = []

    for k, v in payload.items():
        if v is None:
            continue

        val = str(v).strip()

        if val == "" or val.lower() == "nan":
            continue

        parts.append(f"{k}: {val}")

    return " | ".join(parts)[:max_chars]


# =========================
# CLEAN PAYLOAD
# =========================
def clean_payload(payload: dict):
    clean = {}

    for k, v in payload.items():

        if isinstance(v, Image.Image):
            continue

        if hasattr(v, "tolist"):
            v = v.tolist()

        if isinstance(v, bytes):
            v = str(v)

        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)

    return clean


# =========================
# SAVE IMAGE
# =========================
def save_image(image: Image.Image, point_id: str):
    path = os.path.join(IMAGE_DIR, f"{point_id}.jpg")

    try:
        image.convert("RGB").save(path)
        return path
    except Exception as e:
        print(f"❌ Save image error: {e}")
        return None


# =========================
# MAIN
# =========================
def run():
    print("🚀 Start MULTIMODAL ingest (HF dataset)...")

    dataset = load_dataset(
        "pinkieseb/nutrition_dataset",
        split="train",
        streaming=True
    )

    clip = CLIPService()
    qdrant = QdrantService()

    # =========================
    # ENSURE COLLECTIONS
    # =========================
    qdrant.ensure_collection(TEXT_COLLECTION, dim=512)
    qdrant.ensure_collection(IMAGE_COLLECTION, dim=512)

    text_batch = []
    image_batch = []

    BATCH_SIZE = 32
    count = 0

    for item in dataset:

        raw_payload = dict(item)

        # =========================
        # SHARED ID
        # =========================
        point_id = str(uuid.uuid4())

        # =========================
        # CLEAN PAYLOAD
        # =========================
        payload = clean_payload(raw_payload)

        payload["id"] = point_id
        payload["source"] = "hf"
        payload["domain"] = "food"

        # =========================
        # IMAGE PART
        # =========================
        image = raw_payload.get("image", None)

        if isinstance(image, Image.Image):

            image_path = save_image(image, point_id)

            if image_path:
                payload["image_path"] = image_path

                image_vec = clip.embed_image(image_path)

                if image_vec is not None:
                    if hasattr(image_vec, "tolist"):
                        image_vec = image_vec.tolist()

                    image_batch.append({
                        "id": point_id,
                        "vector": image_vec,
                        "payload": payload
                    })

        # =========================
        # TEXT PART
        # =========================
        text = serialize_row(payload)

        if text:
            text_vec = clip.embed_text(text)

            if text_vec is not None:
                if hasattr(text_vec, "tolist"):
                    text_vec = text_vec.tolist()

                text_batch.append({
                    "id": point_id,
                    "vector": text_vec,
                    "payload": payload
                })

        count += 1

        # =========================
        # FLUSH
        # =========================
        if len(text_batch) >= BATCH_SIZE:
            qdrant.upsert_generic(TEXT_COLLECTION, text_batch)
            print(f"📝 Text inserted: {count}")
            text_batch.clear()

        if len(image_batch) >= BATCH_SIZE:
            qdrant.upsert_generic(IMAGE_COLLECTION, image_batch)
            print(f"🖼 Image inserted: {count}")
            image_batch.clear()

        # 🔥 TEST LIMIT (optional)
        # if count > 5000:
        #     break

    # =========================
    # FINAL FLUSH
    # =========================
    if text_batch:
        qdrant.upsert_generic(TEXT_COLLECTION, text_batch)

    if image_batch:
        qdrant.upsert_generic(IMAGE_COLLECTION, image_batch)

    print("✅ DONE MULTIMODAL INGEST")


if __name__ == "__main__":
    run()