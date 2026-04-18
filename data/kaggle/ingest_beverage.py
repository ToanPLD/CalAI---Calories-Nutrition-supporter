import kagglehub
import pandas as pd

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row


def run():
    print("🚀 Start ingest beverage...")

    # tải dataset về local
    dataset_path = kagglehub.dataset_download(
        "heitornunes/caffeine-content-of-drinks"
    )

    print(f"Dataset path: {dataset_path}")

    # 🔥 chọn file đúng (QUAN TRỌNG)
    import os

    files = os.listdir(dataset_path)
    print("Files:", files)

    # ví dụ chọn file đầu tiên
    file_path = os.path.join(dataset_path, files[0])

    # load bằng pandas
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} rows")

    clip = CLIPService()
    qdrant = QdrantService()

    batch = []

    for i, (_, row) in enumerate(df.iterrows()):
        payload = row.to_dict()

        text = serialize_row(payload)
        vector = clip.embed_text(text)

        batch.append({
            "vector": vector,
            "payload": payload
        })

        # batch insert
        if len(batch) >= 128:
            qdrant.upsert_generic("beverage_vectors", batch)
            batch.clear()
            print(f"Inserted {i+1} rows")

    if batch:
        qdrant.upsert_generic("beverage_vectors", batch)

    print("✅ Done ingest beverage")


if __name__ == "__main__":
    run()