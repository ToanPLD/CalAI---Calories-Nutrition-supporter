import kagglehub
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe


def run():
    print("🚀 ingest_beverage")

    dataset_path = kagglehub.dataset_download(
        "utsavdey1410/food-nutrition-dataset"
    )

    files = find_all_csv_files(dataset_path)

    clip = CLIPService()
    qdrant = QdrantService()

    batch = []

    for file in files:
        df = load_csv_safe(file)

        if df is None:
            continue

        for _, row in df.iterrows():
            payload = row.to_dict()
            payload["domain"] = "food"

            text = serialize_row(payload)
            vector = clip.embed_text(text)

            if vector is None:
                continue

            batch.append({
                "vector": vector.tolist(),
                "payload": payload
            })

    qdrant.upsert_generic("food_vectors", batch)


if __name__ == "__main__":
    run()