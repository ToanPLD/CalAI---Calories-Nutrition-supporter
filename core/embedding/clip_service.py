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

        vector = self.model.get_image_features(**inputs)[0].detach().cpu().numpy().tolist()

        self.cache.set(f"img:{img_hash}", vector)

        return vector

    def embed_image_pil(self, image):
        return self.embed_image(image)
    def embed_text(self, text):

        text = text[:300]

        key = f"text:{self._hash_text(text)}"

    
        cached = self.cache.get(key)
        cached_vector = self._decode_cached_vector(cached)
        if cached_vector:
            return cached_vector

        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=77, padding=True).to(self.device)
        vector = self.model.get_text_features(**inputs)[0].detach().cpu().numpy().tolist()

    
        self.cache.set(key, vector)

        return vector
    def embed_text_batch(self, texts):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)

        return outputs.detach().cpu().numpy().tolist()
