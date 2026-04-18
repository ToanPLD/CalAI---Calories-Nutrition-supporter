import os
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


def run():
    print("🚀 Start ingest exercise...")

    # =========================
    # DOWNLOAD DATASET
    # =========================
    dataset_path = kagglehub.dataset_download(
        "aadhavvignesh/calories-burned-during-exercise-and-activities"
    )

    print(f"Dataset path: {dataset_path}")
    print("Files:", os.listdir(dataset_path))

    # =========================
    # FIND CSV FILE (ROBUST)
    # =========================
    file_path = None

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(".csv"):
                file_path = os.path.join(root, f)
                break
        if file_path:
            break

    if file_path is None:
        raise ValueError("❌ No CSV file found")

    print(f"📂 Using file: {file_path}")

    # =========================
    # READ CSV (FULL DATA SAFE)
    # =========================
    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        engine="python",        # 🔥 flexible parser
        on_bad_lines="warn"     # 🔥 không mất data
    )

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # =========================
    # INIT SERVICES
    # =========================
    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    BATCH_SIZE = 64   # 🔥 giảm để ổn định

    # =========================
    # PROCESS
    # =========================
    for idx, row in df.iterrows():

        payload = row.to_dict()

        # 🔥 debug nếu row lỗi
        if len(payload.keys()) < 3:
            print("⚠️ Suspicious row:", payload)

        # =========================
        # SERIALIZE (ANTI CLIP OVERFLOW)
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

        vector = vector.tolist() if hasattr(vector, "tolist") else vector

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # =========================
        # UPSERT BATCH
        # =========================
        if len(batch) >= BATCH_SIZE:
            qdrant.upsert_generic("exercise_vectors", batch)
            print(f"✅ Inserted {idx + 1}")
            batch.clear()

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic("exercise_vectors", batch)

    print("✅ Done ingest exercise")


if __name__ == "__main__":
    run()