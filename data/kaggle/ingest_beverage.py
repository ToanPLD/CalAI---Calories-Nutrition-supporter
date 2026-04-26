# data/kaggle/ingest_beverage.py

import kagglehub

from core.services.retrieval.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe

from core.services.cache.embedding_cache import EmbeddingCache
from core.services.cache.dedup_service import DedupService
from core.embedding.text_embedding_service import TextEmbeddingService


def run():
    print("🚀 ingest_beverage (768 model)")

    dataset_path = kagglehub.dataset_download(
        "heitornunes/caffeine-content-of-drinks"
    )

    files = find_all_csv_files(dataset_path)

    qdrant = QdrantService()
    text_embed = TextEmbeddingService()

    cache = EmbeddingCache()
    dedup = DedupService()

    COLLECTION = "beverage_text_vectors_768"

    BATCH_SIZE = 64
    batch = []
    total = 0

    for file in files:

        print(f"\n📂 Processing: {file}")

        df = load_csv_safe(file)

        if df is None:
            continue

        print(f"📊 Rows: {len(df)}")

        for _, row in df.iterrows():

            payload = row.to_dict()
            payload["domain"] = "beverage"

            if dedup.is_duplicate(payload):
                continue

            text = serialize_row(payload)

            vector = cache.get(text)

            if vector is None:
                vector = text_embed.embed(text)

                if vector is None:
                    continue

                cache.set(text, vector)

            batch.append({
                "vector": vector if isinstance(vector, list) else vector.tolist(),
                "payload": payload
            })

            total += 1

            if len(batch) >= BATCH_SIZE:
                qdrant.upsert_generic(COLLECTION, batch)
                print(f"✅ Inserted: {total}")
                batch.clear()

    if batch:
        qdrant.upsert_generic(COLLECTION, batch)

    print(f"\n🎯 DONE: {total} records inserted")

if __name__ == "__main__":
    run()