import uuid
import pandas as pd
import kagglehub

from core.embedding.clip_service import CLIPService
from core.services.qdrant_meta_service import QdrantMetaService
from data.kaggle.utils import find_all_csv_files, load_csv_safe


clip = CLIPService()
meta = QdrantMetaService()


# =========================
# BUILD META ITEM
# =========================
def build_meta_item(payload, domain):

    text = (
        payload.get("Food_Item")
        or payload.get("drink")
        or payload.get("Activity")
        or payload.get("name")
        or payload.get("Name")
        or str(payload)
    )

    vector = clip.embed_text(text)

    return {
        "id": uuid.uuid4().int >> 64,
        "vector": vector,
        "payload": {
            "name": text,
            "domain": str(domain),
            "ref_id": payload.get("__id")
        }
    }


# =========================
# RUN INGEST
# =========================
def run(dataset_name, domain):

    print(f"🚀 Ingest META: {domain}")

    if not isinstance(domain, str):
        raise ValueError(f"❌ Invalid domain: {domain}")

    dataset_path = kagglehub.dataset_download(dataset_name)
    files = find_all_csv_files(dataset_path)

    if not files:
        raise ValueError("❌ No CSV files found")

    items = []
    total = 0

    for file in files:

        print(f"📂 Processing: {file}")

        df = load_csv_safe(file)

        if df is None:
            continue

        for idx, row in df.iterrows():
            payload = row.to_dict()
            payload["__id"] = idx

            item = build_meta_item(payload, domain)
            items.append(item)
            total += 1

            if len(items) >= 64:
                meta.upsert_meta(items)
                print(f"✅ Inserted: {total}")
                items.clear()

    if items:
        meta.upsert_meta(items)

    print(f"🎯 DONE META: {total} records ({domain})")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # 👉 CHỌN DATASET + DOMAIN Ở ĐÂY
    run(
        dataset_name="heitornunes/caffeine-content-of-drinks",
        domain="beverage"
    )

    run(
        dataset_name="adilshamim8/daily-food-and-nutrition-dataset",
        domain="food"
    )

    run(
        dataset_name="jockeroika/calories-burned",
        domain="exercise"
    )

    run(
        dataset_name="ziya07/diet-recommendations-dataset",
        domain="diet"
    )

    run(
        dataset_name="aadhavvignesh/calories-burned-during-exercise-and-activities",
        domain="exercise"
    )
    run(
        dataset_name="joebeachcapital/fast-food",
        domain="food"
    )
    run(
        dataset_name="utsavdey1410/food-nutrition-dataset",
        domain="food"
    )
    run(
        dataset_name="thedevastator/the-nutritional-content-of-food-a-comprehensive",
        domain="food"
    )
    run(
        dataset_name="suvidyasonawane/fruits-nutrition-datasets",
        domain="food"
    )
    run(
        dataset_name="kanchana1990/global-food-nutrition-database10k-products",
        domain="food"
    )
    run(
        dataset_name="valakhorasani/gym-members-exercise-dataset",
        domain="exercise"
    )
    run(
        dataset_name="jockeroika/life-style-data",
        domain="lifestyle"
    )
    run(
        dataset_name="trolukovich/nutritional-values-for-common-foods-and-products",
        domain="food"
    )
    run(
        dataset_name="fatemehmehrparvar/obesity-levels",
        domain="lifestyle"
    )
    run(
        dataset_name="thedevastator/better-recipes-for-a-better-life",
        domain="recipe"
    )
    run(
        dataset_name="henryshan/starbucks",
        domain="beverage"
    )
    run(
        dataset_name="prajwaldongre/collection-of-recipes-around-the-world",
        domain="recipe"
    )

    
