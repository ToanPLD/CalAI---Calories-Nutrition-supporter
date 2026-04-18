import os
import csv
import pandas as pd
import kagglehub

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


# =========================
# 🔥 SAFE CSV LOADER (KHÔNG MẤT DATA)
# =========================
def load_csv_full_safe(file_path):
    rows = []
    max_cols = 0

    # đọc raw CSV
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)

        for row in reader:
            rows.append(row)
            if len(row) > max_cols:
                max_cols = len(row)

    print(f"Max columns detected: {max_cols}")

    # pad None cho dòng thiếu cột
    normalized_rows = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [None] * (max_cols - len(row))
        normalized_rows.append(row)

    # header
    header = normalized_rows[0]

    fixed_header = []
    for i, col in enumerate(header):
        if not col or str(col).strip() == "":
            fixed_header.append(f"unknown_col_{i}")
        else:
            fixed_header.append(str(col).strip())

    df = pd.DataFrame(normalized_rows[1:], columns=fixed_header)

    return df


# =========================
# MAIN
# =========================
def run():
    print("🚀 Start ingest calories_burned...")

    # =========================
    # LOAD DATASET
    # =========================
    dataset_path = kagglehub.dataset_download(
        "jockeroika/calories-burned"
    )

    print(f"Dataset path: {dataset_path}")
    print("Files:", os.listdir(dataset_path))

    file_path = None
    for f in os.listdir(dataset_path):
        if f.lower().endswith(".csv"):
            file_path = os.path.join(dataset_path, f)
            break

    if file_path is None:
        raise ValueError("❌ No CSV file found")

    # 🔥 dùng SAFE loader
    df = load_csv_full_safe(file_path)

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

        # debug nếu thiếu field
        if len(payload.keys()) < 5:
            print("⚠️ Suspicious row:", payload)

        # serialize full data
        text = serialize_row(payload)

        vector = clip.embed_text(text)

        if vector is None:
            continue

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # =========================
        # UPSERT
        # =========================
        if len(batch) >= batch_size:
            qdrant.upsert_generic("exercise_vectors", batch)
            batch.clear()
            print(f"✅ Upserted {idx + 1}")

    # flush cuối
    if batch:
        qdrant.upsert_generic("exercise_vectors", batch)

    print("✅ Done ingest calories_burned")


if __name__ == "__main__":
    run()