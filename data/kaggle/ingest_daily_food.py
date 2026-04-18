import os
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


def run():
    print("🚀 Start ingest daily_food...")

    # =========================
    # DOWNLOAD DATASET
    # =========================
    dataset_path = kagglehub.dataset_download(
        "adilshamim8/daily-food-and-nutrition-dataset"
    )

    print(f"Dataset path: {dataset_path}")
    print("Files:", os.listdir(dataset_path))

    # =========================
    # FIND CSV FILE
    # =========================
    file_path = None
    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            file_path = os.path.join(dataset_path, f)
            break

    if file_path is None:
        raise ValueError("❌ No CSV file found")

    # =========================
    # READ CSV (FULL DATA SAFE)
    # =========================
    df = pd.read_csv(
    file_path,
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"   # hoặc "warn"
)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # =========================
    # INIT SERVICES
    # =========================
    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    batch_size = 64

    # =========================
    # PROCESS
    # =========================
    for idx, row in df.iterrows():

        payload = row.to_dict()

        # 🔥 đảm bảo không mất data
        if len(payload.keys()) < 5:
            print("⚠️ Suspicious row:", payload)

        # =========================
        # TEXT BUILD
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

        # numpy → list
        vector = vector.tolist() if hasattr(vector, "tolist") else vector

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # =========================
        # UPSERT BATCH
        # =========================
        if len(batch) >= batch_size:
            qdrant.upsert_generic("food_text_vectors", batch)
            batch.clear()
            print(f"✅ Upserted {idx + 1}")

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic("food_text_vectors", batch)

    print("✅ Done ingest daily_food")


if __name__ == "__main__":
    run()