import torch
import hashlib
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from core.services.cache.redis_cache import RedisCache


class CLIPService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.model.eval()

        self.cache = RedisCache()

    def _hash_text(self, text: str):
        text = text.lower().strip()
        return hashlib.md5(text.encode()).hexdigest()

    def _decode_cached_vector(self, cached):
        if cached is None:
            return None
        if isinstance(cached, list):
            return cached
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")
        if isinstance(cached, str):
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                return None
        return None

    def _hash_image(self, image):
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return hashlib.md5(buffer.getvalue()).hexdigest(), buffer.getvalue()

    def embed_image(self, image):

        img_hash, _ = self._hash_image(image)

        cached = self.cache.get(f"img:{img_hash}")
        cached_vector = self._decode_cached_vector(cached)
        if cached_vector:
            return cached_vector

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vector = self.model.get_image_features(**inputs)[0].detach().cpu().numpy().tolist()

        self.cache.set(f"img:{img_hash}", vector)

        return vector

    def embed_image_pil(self, image):
        return self.embed_image(image)

    def embed_images_batch(self, images):
        if not images:
            return []

        vectors = [None] * len(images)
        missing = []
        missing_indexes = []

        for index, image in enumerate(images):
            img_hash, _ = self._hash_image(image)
            cached = self.cache.get(f"img:{img_hash}")
            cached_vector = self._decode_cached_vector(cached)
            if cached_vector:
                vectors[index] = cached_vector
                continue

            missing.append((img_hash, image))
            missing_indexes.append(index)

        if missing:
            inputs = self.processor(
                images=[image for _, image in missing],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                batch_vectors = (
                    self.model
                    .get_image_features(**inputs)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            for index, (img_hash, _), vector in zip(
                missing_indexes,
                missing,
                batch_vectors
            ):
                self.cache.set(f"img:{img_hash}", vector)
                vectors[index] = vector

        return vectors

    def embed_text(self, text):

        text = text[:300]

        key = f"text:{self._hash_text(text)}"

    
        cached = self.cache.get(key)
        cached_vector = self._decode_cached_vector(cached)
        if cached_vector:
            return cached_vector

        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=77, padding=True).to(self.device)
        with torch.no_grad():
            vector = self.model.get_text_features(**inputs)[0].detach().cpu().numpy().tolist()

    
        self.cache.set(key, vector)

        return vector
    def embed_text_batch(self, texts):

        vectors = []
        batch_size = 64

        for start in range(0, len(texts), batch_size):
            batch = [text[:300] for text in texts[start:start + batch_size]]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)

            vectors.extend(outputs.detach().cpu().numpy().tolist())

        return vectors
