# data/kaggle/ingest_fastfood.py

import kagglehub

from core.services.retrieval.qdrant_service import QdrantService
from core.utils.text_serializer import serialize_row
from data.kaggle.utils import find_all_csv_files, load_csv_safe

from core.services.cache.embedding_cache import EmbeddingCache
from core.services.cache.dedup_service import DedupService
from core.embedding.text_embedding_service import TextEmbeddingService


def run():
    print("🚀 ingest_fastfood (768 model)")

    dataset_path = kagglehub.dataset_download(
        "joebeachcapital/fast-food"
    )

    files = find_all_csv_files(dataset_path)

    qdrant = QdrantService()
    text_embed = TextEmbeddingService()

    cache = EmbeddingCache()
    dedup = DedupService()

    COLLECTION = "food_vectors_768"

    UPSERT_BATCH = 64
    EMBED_BATCH = 64

    batch = []
    texts = []
    payloads = []

    total = 0

    for file in files:

        print(f"\n📂 Processing: {file}")

        df = load_csv_safe(file)
        if df is None:
            continue

        print(f"📊 Rows: {len(df)}")

        for _, row in df.iterrows():

            payload = row.to_dict()
            payload["domain"] = "food"

            # =========================
            # DEDUP
            # =========================
            if dedup.is_duplicate(payload):
                continue

            text = serialize_row(payload)

            # 🔥 cache riêng cho 768
            cache_key = f"bge768:{text}"
            cached = cache.get(cache_key)

            if cached:
                batch.append({
                    "vector": cached,
                    "payload": payload
                })
                total += 1
                continue

            texts.append(text)
            payloads.append(payload)

            # =========================
            # BATCH EMBEDDING
            # =========================
            if len(texts) >= EMBED_BATCH:

                vectors = text_embed.embed_batch(texts)

                for t, p, v in zip(texts, payloads, vectors):

                    if v is None:
                        continue

                    cache.set(f"bge768:{t}", v)

                    batch.append({
                        "vector": v,
                        "payload": p
                    })

                    total += 1

                texts.clear()
                payloads.clear()

            # =========================
            # UPSERT
            # =========================
            if len(batch) >= UPSERT_BATCH:
                qdrant.upsert_generic(COLLECTION, batch)
                print(f"✅ Inserted: {total}")
                batch.clear()

    # =========================
    # FINAL FLUSH
    # =========================
    if texts:
        vectors = text_embed.embed_batch(texts)

        for t, p, v in zip(texts, payloads, vectors):

            if v is None:
                continue

            cache.set(f"bge768:{t}", v)

            batch.append({
                "vector": v,
                "payload": p
            })

            total += 1

    if batch:
        qdrant.upsert_generic(COLLECTION, batch)

    print(f"\n🎯 DONE ingest_fastfood → {total} records")


if __name__ == "__main__":
    run()