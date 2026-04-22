import kagglehub
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe


def run():
    print("🚀 ingest_food_nutrition")

    dataset = "thedevastator/better-recipes-for-a-better-life"
    collection = "recipes_vectors"
    domain = "recipe"

    BATCH_SIZE = 16 

    dataset_path = kagglehub.dataset_download(dataset)
    files = find_all_csv_files(dataset_path)

    clip = CLIPService()
    qdrant = QdrantService()

    batch = []
    total = 0

    for file in files:
        df = load_csv_safe(file)

        if df is None:
            continue

        for idx, row in df.iterrows():

            payload = row.to_dict()
            payload["domain"] = domain

            for k, v in payload.items():
                if isinstance(v, str) and len(v) > 120:
                    payload[k] = v[:120]

            text = serialize_row(payload)

            vector = clip.embed_text(text)

            if vector is None:
                continue

            batch.append({
                "vector": vector.tolist(),
                "payload": payload
            })

            total += 1

            if len(batch) >= BATCH_SIZE:
                qdrant.upsert_generic(collection, batch)
                print(f"✅ Upserted {total}")
                batch.clear()

    if batch:
        qdrant.upsert_generic(collection, batch)

    print("✅ Done ingest_food_nutrition")


if __name__ == "__main__":
    run()