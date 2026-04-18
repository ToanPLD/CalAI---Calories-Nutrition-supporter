import os
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


# =========================
# FIND FILE (RECURSIVE)
# =========================
def find_data_file(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".csv", ".xlsx", ".xls")):
                return os.path.join(root, f)
    return None


# =========================
# LOAD DATAFRAME
# =========================
def load_dataframe(file_path):

    if file_path.endswith(".csv"):
        return pd.read_csv(
            file_path,
            encoding="utf-8",
            engine="python",
            on_bad_lines="warn"   # 🔥 giữ max data
        )

    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported file: {file_path}")


# =========================
# MAIN
# =========================
def run():
    print("🚀 Start ingest food nutrition (ABBREV dataset)...")

    # =========================
    # DOWNLOAD
    # =========================
    dataset_path = kagglehub.dataset_download(
        "thedevastator/the-nutritional-content-of-food-a-comprehensive"
    )

    print(f"Dataset path: {dataset_path}")
    print("Root files:", os.listdir(dataset_path))

    # =========================
    # FIND FILE
    # =========================
    file_path = find_data_file(dataset_path)

    if not file_path:
        raise ValueError("❌ No data file found")

    print(f"📂 Using file: {file_path}")

    # =========================
    # LOAD
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

        # 🔥 FULL payload (vitamin cực nhiều)
        payload = row.to_dict()

        # sanity check
        if len(payload.keys()) < 10:
            print("⚠️ Suspicious row:", payload)

        # =========================
        # SERIALIZE
        # =========================
        text = serialize_row(payload)

        if not text:
            continue

        # =========================
        # EMBED
        # =========================
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

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic("food_text_vectors", batch)

    print("✅ Done ingest food nutrition dataset")


if __name__ == "__main__":
    run()