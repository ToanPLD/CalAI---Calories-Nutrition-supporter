import os
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


def run():
    print("🚀 Start ingest fast-food...")

    # =========================
    # DOWNLOAD DATASET
    # =========================
    dataset_path = kagglehub.dataset_download(
        "joebeachcapital/fast-food"
    )

    print(f"Dataset path: {dataset_path}")
    print("Files:", os.listdir(dataset_path))

    # =========================
    # AUTO DETECT FILE
    # =========================
    file_path = None

    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            file_path = os.path.join(dataset_path, f)
            break

    if file_path is None:
        raise ValueError("❌ No CSV file found")

    print(f"📂 Using file: {file_path}")

    # =========================
    # LOAD DATA (FULL, KHÔNG MẤT DATA)
    # =========================
    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        engine="python",        # 🔥 tránh crash
        on_bad_lines="warn"     # 🔥 giữ data tối đa
    )

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # =========================
    # INIT SERVICES
    # =========================
    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    BATCH_SIZE = 64

    # =========================
    # PROCESS
    # =========================
    for idx, row in df.iterrows():

        # 🔥 giữ toàn bộ dữ liệu
        payload = row.to_dict()

        # =========================
        # SERIALIZE TEXT
        # =========================
        text = serialize_row(payload)

        if not text or len(text.strip()) == 0:
            continue

        # =========================
        # EMBEDDING
        # =========================
        vector = clip.embed_text(text)

        if vector is None:
            continue

        # convert numpy → list
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # =========================
        # UPSERT BATCH
        # =========================
        if len(batch) >= BATCH_SIZE:
            qdrant.upsert_generic("food_text_vectors", batch)
            print(f"✅ Inserted {idx + 1}")
            batch.clear()

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic("food_text_vectors", batch)

    print("✅ Done ingest fast-food")


if __name__ == "__main__":
    run()