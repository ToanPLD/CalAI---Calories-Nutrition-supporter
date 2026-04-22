# data/kaggle/ingest_daily_food.py

import kagglehub
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row

from data.kaggle.utils import find_all_csv_files, load_csv_safe


def run():
    print("🚀 ingest_daily_food")

    # =========================
    # DOWNLOAD DATASET
    # =========================
    dataset_path = kagglehub.dataset_download(
        "adilshamim8/daily-food-and-nutrition-dataset"
    )

    # =========================
    # LOAD ALL CSV FILES
    # =========================
    csv_files = find_all_csv_files(dataset_path)

    if not csv_files:
        raise ValueError("❌ No CSV found")

    df_list = []

    for f in csv_files:
        df_list.append(load_csv_safe(f))

    import pandas as pd
    df = pd.concat(df_list, ignore_index=True)

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

        payload = row.to_dict()

        text = serialize_row(payload)
        vector = clip.embed_text(text)

        if vector is None:
            continue

        batch.append({
            "vector": vector,
            "payload": payload
        })

        if len(batch) >= BATCH_SIZE:
            qdrant.upsert_points("food_text_vectors", batch)
            print(f"✅ Upserted {idx + 1}")
            batch.clear()

    # final flush
    if batch:
        qdrant.upsert_points("food_text_vectors", batch)

    print("✅ DONE ingest_daily_food")


if __name__ == "__main__":
    run()