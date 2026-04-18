import os
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


def find_data_file(dataset_path):
    """
    🔥 Tìm file dataset trong folder (recursive)
    """
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".csv", ".xlsx", ".xls")):
                return os.path.join(root, f)

    return None


def load_dataframe(file_path):
    """
    🔥 Load file linh hoạt
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(
            file_path,
            encoding="utf-8",
            engine="python",
            on_bad_lines="warn"
        )

    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported file: {file_path}")


def run():
    print("🚀 Start ingest food FULL nutrition...")

    # =========================
    # DOWNLOAD
    # =========================
    dataset_path = kagglehub.dataset_download(
        "utsavdey1410/food-nutrition-dataset"
    )

    print(f"Dataset path: {dataset_path}")
    print("Root files:", os.listdir(dataset_path))

    # =========================
    # FIND FILE (FIX CHÍNH)
    # =========================
    file_path = find_data_file(dataset_path)

    if not file_path:
        raise ValueError("❌ No data file found inside dataset")

    print(f"📂 Using file: {file_path}")

    # =========================
    # LOAD DATA
    # =========================
    df = load_dataframe(file_path)

    print(f"Loaded {len(df)} rows")
    print(f"Columns count: {len(df.columns)}")

    # =========================
    # INIT
    # =========================
    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    BATCH_SIZE = 64

    # =========================
    # PROCESS
    # =========================
    for idx, row in df.iterrows():

        payload = row.to_dict()

        text = serialize_row(payload)

        if not text:
            continue

        vector = clip.embed_text(text)

        if vector is None:
            continue

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # =========================
        # UPSERT
        # =========================
        if len(batch) >= BATCH_SIZE:
            qdrant.upsert_generic("food_text_vectors", batch)
            print(f"✅ Inserted {idx + 1}")
            batch.clear()

    # final flush
    if batch:
        qdrant.upsert_generic("food_text_vectors", batch)

    print("✅ Done ingest food FULL nutrition")


if __name__ == "__main__":
    run()