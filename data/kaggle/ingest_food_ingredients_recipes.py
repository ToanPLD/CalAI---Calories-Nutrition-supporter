import argparse
import ast
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

import kagglehub
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from core.embedding.clip_service import CLIPService
from core.embedding.text_embedding_service import TextEmbeddingService
from core.services.cache.embedding_cache import EmbeddingCache
from core.services.qdrant_meta_service import QdrantMetaService
from core.services.rag.recipe_image_rag_service import RecipeImageRAGService
from core.services.vision.qwen_vl_service import QwenVLService
from data.kaggle.utils import find_all_csv_files, load_csv_safe


DATASET = settings.RECIPE_IMAGE_DATASET
COLLECTION = settings.RECIPE_IMAGE_DATASET_COLLECTION
DOMAIN = settings.RECIPE_IMAGE_META_DOMAIN

REQUIRED_COLUMNS = {
    "Title",
    "Ingredients",
    "Instructions",
    "Image_Name",
    "Cleaned_Ingredients",
}


def stable_point_id(*parts):
    raw = "::".join(str(part or "") for part in parts)
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:16], 16)


def safe_str(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def slugify(value):
    value = safe_str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_list_field(value):
    text = safe_str(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [
                safe_str(item)
                for item in parsed
                if safe_str(item)
            ]
    except Exception:
        pass

    return [
        item.strip(" -")
        for item in re.split(r"\s*[;,]\s*", text)
        if item.strip(" -")
    ]


def find_recipe_csv(dataset_path):
    for csv_file in find_all_csv_files(dataset_path):
        try:
            sample = pd.read_csv(csv_file, nrows=2)
        except Exception:
            continue

        columns = {column.strip() for column in sample.columns}
        if REQUIRED_COLUMNS.issubset(columns):
            return csv_file

    raise ValueError(
        "No CSV with Title, Ingredients, Instructions, Image_Name, "
        "Cleaned_Ingredients was found."
    )


def build_image_index(dataset_path):
    image_index = {}
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for root, _, files in os.walk(dataset_path):
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() not in extensions:
                continue

            image_index[path.stem.lower()] = str(path)
            image_index[filename.lower()] = str(path)

    print(f"🖼️ Indexed {len(image_index)} image lookup keys")
    return image_index


def resolve_image_path(image_index, image_name):
    image_name = safe_str(image_name)
    if not image_name:
        return None

    key = image_name.lower()
    if key in image_index:
        return image_index[key]

    stem = Path(image_name).stem.lower()
    return image_index.get(stem)


def build_metadata_caption(title, ingredients, image_name):
    top_ingredients = ", ".join(ingredients[:8])
    if top_ingredients:
        return f"Recipe image for {title}. Key ingredients: {top_ingredients}."
    return f"Recipe image for {title or image_name or 'unknown dish'}."


def caption_with_vision(qwen, cache, image_path, title, ingredients, image_name):
    cache_key = f"recipe_image_caption:{image_name}:{title}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((512, 512))
        caption = asyncio.run(
            qwen.caption_food_image(
                image=image,
                filename_hint=image_name,
                title=title,
                ingredients=ingredients[:12]
            )
        )
        cache.set(cache_key, caption)
        return caption
    except Exception as exc:
        return {
            "caption": None,
            "visual_tags": [],
            "visible_ingredients": [],
            "confidence": 0,
            "error": str(exc)
        }


def build_payload(row, row_index, dataset_path, image_index, caption_mode, qwen, cache):
    title = safe_str(row.get("Title"))
    image_name = safe_str(row.get("Image_Name"))
    ingredients_raw = safe_str(row.get("Ingredients"))
    cleaned_raw = safe_str(row.get("Cleaned_Ingredients"))
    instructions = safe_str(row.get("Instructions"))
    ingredients_list = parse_list_field(ingredients_raw)
    cleaned_ingredients = parse_list_field(cleaned_raw) or ingredients_list
    image_path = resolve_image_path(image_index, image_name)
    image_extension = Path(image_path).suffix.lower().lstrip(".") if image_path else None
    metadata_caption = build_metadata_caption(
        title=title,
        ingredients=cleaned_ingredients,
        image_name=image_name
    )

    vision_caption = {}
    if caption_mode == "vision" and image_path:
        vision_caption = caption_with_vision(
            qwen=qwen,
            cache=cache,
            image_path=image_path,
            title=title,
            ingredients=cleaned_ingredients,
            image_name=image_name
        )

    if caption_mode == "none":
        image_caption = None
    else:
        image_caption = (
            safe_str(vision_caption.get("caption"))
            if vision_caption.get("caption")
            else metadata_caption
        )

    text_for_search = " | ".join([
        f"Title: {title}",
        f"Ingredients: {', '.join(cleaned_ingredients[:40])}",
        f"Instructions: {instructions[:900]}",
        f"Image caption: {image_caption or ''}",
    ])[:1800]

    point_id = stable_point_id(DATASET, row_index, image_name, title)

    return {
        "point_id": point_id,
        "text_for_search": text_for_search,
        "image_vector_text": " | ".join([
            image_caption or "",
            title,
            ", ".join(cleaned_ingredients[:20])
        ]).strip(),
        "image_path": image_path,
        "payload": {
            "domain": DOMAIN,
            "source_dataset": DATASET,
            "source_collection": COLLECTION,
            "source_row": int(row_index),
            "source_csv": "Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
            "dataset_path": str(dataset_path),
            "title": title,
            "title_slug": slugify(title),
            "name": title,
            "dish_name": title,
            "recipe_title": title,
            "ingredients": ingredients_raw,
            "ingredients_list": ingredients_list,
            "cleaned_ingredients": cleaned_raw,
            "cleaned_ingredients_list": cleaned_ingredients,
            "ingredients_search": " ".join(cleaned_ingredients).lower(),
            "ingredient_count": len(cleaned_ingredients),
            "instructions": instructions,
            "instructions_preview": instructions[:700],
            "image_name": image_name,
            "image_file": os.path.basename(image_path) if image_path else None,
            "image_path": image_path,
            "image_extension": image_extension,
            "has_image": bool(image_path),
            "image_caption": image_caption,
            "caption_mode": caption_mode,
            "has_caption": bool(image_caption),
            "vision_caption": vision_caption,
            "visual_tags": vision_caption.get("visual_tags", []),
            "visible_ingredients": vision_caption.get("visible_ingredients", []),
            "retrieval_text": text_for_search,
            "citation": {
                "dataset": DATASET,
                "csv": "Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
                "row": int(row_index),
                "title": title,
                "image_name": image_name,
                "image_file": os.path.basename(image_path) if image_path else None,
            }
        }
    }


def cached_text_vectors(text_embed, cache, texts):
    vectors = [None] * len(texts)
    missing = []
    missing_indexes = []

    for index, text in enumerate(texts):
        key = f"recipe_image_text:{text}"
        cached = cache.get(key)
        if cached:
            vectors[index] = cached
            continue
        missing.append(text)
        missing_indexes.append(index)

    if missing:
        encoded = text_embed.embed_batch(missing)
        for index, text, vector in zip(missing_indexes, missing, encoded):
            cache.set(f"recipe_image_text:{text}", vector)
            vectors[index] = vector

    return vectors


def build_image_vectors(clip, payloads, image_vector_source):
    vectors = [None] * len(payloads)

    if image_vector_source == "pixels":
        images = []
        image_indexes = []
        for index, payload in enumerate(payloads):
            image_path = payload.get("image_path")
            if not image_path:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((384, 384))
            except Exception:
                continue

            images.append(image)
            image_indexes.append(index)

        if images:
            for index, vector in zip(image_indexes, clip.embed_images_batch(images)):
                vectors[index] = vector

    text_vector_inputs = []
    text_vector_indexes = []
    for index, payload in enumerate(payloads):
        if vectors[index] is not None:
            continue

        text_vector_inputs.append(payload.get("image_vector_text") or payload["text_for_search"])
        text_vector_indexes.append(index)

    if text_vector_inputs:
        for index, vector in zip(text_vector_indexes, clip.embed_text_batch(text_vector_inputs)):
            vectors[index] = vector

    return vectors


def flush_batch(service, meta, text_embed, clip, cache, pending, upsert_meta):
    if not pending:
        return 0

    texts = [item["text_for_search"] for item in pending]
    text_vectors = cached_text_vectors(text_embed, cache, texts)
    image_vectors = build_image_vectors(
        clip=clip,
        payloads=pending,
        image_vector_source=pending[0].get("image_vector_source", "pixels")
    )

    points = []
    meta_points = []
    for item, text_vector, image_vector in zip(pending, text_vectors, image_vectors):
        if text_vector is None or image_vector is None:
            continue

        points.append({
            "id": item["point_id"],
            "text_vector": text_vector,
            "image_vector": image_vector,
            "payload": item["payload"]
        })

        if upsert_meta:
            meta_points.append({
                "id": stable_point_id("meta", item["point_id"]),
                "vector": text_vector,
                "payload": {
                    "name": item["payload"]["title"],
                    "domain": DOMAIN,
                    "ref_id": item["point_id"],
                    "source_dataset": DATASET,
                    "image_name": item["payload"]["image_name"],
                }
            })

    service.upsert_points(points)
    if upsert_meta and meta_points:
        meta.upsert_meta(meta_points)

    return len(points)


def run(
    limit=None,
    start_row=0,
    batch_size=32,
    caption_mode="metadata",
    image_vector_source="pixels",
    recreate=False,
    upsert_meta=True
):
    print("🚀 Ingest Food Ingredients + Recipe Images")
    print(f"📦 Dataset: {DATASET}")
    print(f"🎯 Collection: {COLLECTION}")

    dataset_path = kagglehub.dataset_download(DATASET)
    csv_file = find_recipe_csv(dataset_path)
    image_index = build_image_index(dataset_path)
    df = load_csv_safe(csv_file)
    if df is None:
        raise ValueError("Could not load recipe CSV.")

    if start_row:
        df = df.iloc[start_row:]
    if limit:
        df = df.head(limit)

    service = RecipeImageRAGService(collection_name=COLLECTION)
    service.ensure_collection(recreate=recreate)
    meta = QdrantMetaService() if upsert_meta else None
    text_embed = TextEmbeddingService()
    clip = CLIPService()
    cache = EmbeddingCache()
    qwen = QwenVLService() if caption_mode == "vision" else None

    pending = []
    total = 0

    for row_index, row in df.iterrows():
        item = build_payload(
            row=row,
            row_index=row_index,
            dataset_path=dataset_path,
            image_index=image_index,
            caption_mode=caption_mode,
            qwen=qwen,
            cache=cache
        )
        item["image_vector_source"] = image_vector_source
        pending.append(item)

        if len(pending) >= batch_size:
            inserted = flush_batch(
                service=service,
                meta=meta,
                text_embed=text_embed,
                clip=clip,
                cache=cache,
                pending=pending,
                upsert_meta=upsert_meta
            )
            total += inserted
            print(f"✅ Upserted {total} recipes")
            pending.clear()

    if pending:
        inserted = flush_batch(
            service=service,
            meta=meta,
            text_embed=text_embed,
            clip=clip,
            cache=cache,
            pending=pending,
            upsert_meta=upsert_meta
        )
        total += inserted

    print(f"🎯 DONE → {total} records in {COLLECTION}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Kaggle food recipe + image dataset into Qdrant."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--caption-mode",
        choices=["metadata", "vision", "none"],
        default="metadata",
        help=(
            "metadata is fast; vision calls Qwen-VL per image and is much slower; "
            "none stores no caption."
        )
    )
    parser.add_argument(
        "--image-vector-source",
        choices=["pixels", "caption"],
        default="pixels",
        help="pixels uses CLIP image embeddings when image files exist."
    )
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-meta", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        limit=args.limit,
        start_row=args.start_row,
        batch_size=args.batch_size,
        caption_mode=args.caption_mode,
        image_vector_source=args.image_vector_source,
        recreate=args.recreate,
        upsert_meta=not args.no_meta
    )
