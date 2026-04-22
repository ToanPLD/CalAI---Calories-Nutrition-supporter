import kagglehub
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe


def run():
    print("🚀 ingest_nutrition_csv")

    # =========================
    # CONFIG
    # =========================
    dataset = "trolukovich/nutritional-values-for-common-foods-and-products"
    collection = "food_vectors"
    domain = "food"

    BATCH_SIZE = 32  # 🔥 giảm xuống

    # =========================
    # LOAD
    # =========================
    dataset_path = kagglehub.dataset_download(dataset)
    files = find_all_csv_files(dataset_path)

    clip = CLIPService()
    qdrant = QdrantService()

    batch = []

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

            text = serialize_row(payload)
            vector = clip.embed_text(text)

            if vector is None:
                continue

            batch.append({
                "vector": vector.tolist(),
                "payload": payload
            })

            # 🔥 FLUSH NHỎ
            if len(batch) >= BATCH_SIZE:
                qdrant.upsert_generic(collection, batch)
                print(f"✅ Upserted {idx}")
                batch.clear()

    # =========================
    # FINAL FLUSH
    # =========================
    if batch:
        qdrant.upsert_generic(collection, batch)

    print("✅ Done ingest_nutrition_csv")


if __name__ == "__main__":
    run()