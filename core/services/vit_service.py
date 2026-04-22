# core/services/vit_service.py

import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from core.utils.logger import get_logger

logger = get_logger("vit")


class ViTService:

    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"ViT loaded on {self.device}")

    @torch.inference_mode()
    def embed(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]

            embedding = F.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy()[0].tolist()

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None