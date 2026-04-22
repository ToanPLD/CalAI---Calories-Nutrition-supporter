import uuid
from core.services.clip_service import CLIPService
from core.services.qdrant_meta_service import QdrantMetaService


clip = CLIPService()
meta = QdrantMetaService()


def build_meta_item(payload, domain):

    text = payload.get("Food_Item") or payload.get("drink") or payload.get("Activity") or ""

    vector = clip.embed_text(text)

    return {
        "id": uuid.uuid4().int >> 64,
        "vector": vector,
        "payload": {
            "name": text,
            "domain": domain,
            "ref_id": payload.get("id", uuid.uuid4().int >> 64)
        }
    }


def run(dataset, domain):

    items = []

    for _, row in dataset.iterrows():
        payload = row.to_dict()

        item = build_meta_item(payload, domain)

        items.append(item)

        if len(items) >= 64:
            meta.upsert_meta(items)
            items.clear()

    if items:
        meta.upsert_meta(items)

    print(f"✅ Meta ingest done: {domain}") 