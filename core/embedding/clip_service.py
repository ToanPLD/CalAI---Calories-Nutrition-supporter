import torch
import hashlib
import numpy as np
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

    def _hash_image(self, image):
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return hashlib.md5(buffer.getvalue()).hexdigest(), buffer.getvalue()

    def embed_image(self, image):

        img_hash, img_bytes = self._hash_image(image)

        cached = self.cache.client.get(f"img:{img_hash}")
        if cached:
            return np.frombuffer(cached, dtype=np.float32).tolist()

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        vector = self.model.get_image_features(**inputs)[0].detach().cpu().numpy()

        self.cache.client.setex(
            f"img:{img_hash}",
            self.cache.ttl,
            vector.astype(np.float32).tobytes()
        )

        return vector.tolist()

    def embed_image_pil(self, image):
        return self.embed_image(image)
    def embed_text(self, text):

        text = text[:300]

        key = f"text:{text}"

    
        cached = self.cache.get(key)
        if cached:
            return cached

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