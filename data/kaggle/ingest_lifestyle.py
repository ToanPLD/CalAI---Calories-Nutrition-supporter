import kagglehub
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe


def run():
    print("🚀 ingest_lifestyle")

    # =========================
    # CONFIG
    # =========================
    dataset = "jockeroika/life-style-data"
    collection = "lifestyle_vectors"
    domain = "lifestyle"

    BATCH_SIZE = 16   # 🔥 dataset này PHẢI nhỏ

    # =========================
    # LOAD DATA
    # =========================
    dataset_path = kagglehub.dataset_download(dataset)
    files = find_all_csv_files(dataset_path)

    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    total = 0

    # =========================
    # PROCESS
    # =========================
    for file in files:
        df = load_csv_safe(file)

        if df is None:
            continue

        for idx, row in df.iterrows():

            payload = row.to_dict()
            payload["domain"] = domain

            # 🔥 giảm payload size (QUAN TRỌNG)
            for k, v in payload.items():
                if isinstance(v, str) and len(v) > 150:
                    payload[k] = v[:150]

            text = serialize_row(payload)

            vector = clip.embed_text(text)

            if vector is None:
                continue

            batch.append({
                "vector": vector.tolist(),
                "payload": payload
            })

            total += 1

            # =========================
            # FLUSH NHỎ
            # =========================
            if len(batch) >= BATCH_SIZE:
                qdrant.upsert_generic(collection, batch)
                print(f"✅ Upserted {total}")
                batch.clear()

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic(collection, batch)

    print("✅ Done ingest_lifestyle")


if __name__ == "__main__":
    run()