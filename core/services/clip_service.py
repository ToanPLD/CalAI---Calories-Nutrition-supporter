# core/services/clip_service.py

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from core.utils.logger import get_logger

logger = get_logger("clip")


class CLIPService:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"🚀 CLIP running on {self.device}")

    # =========================
    # IMAGE EMBEDDING
    # =========================
    @torch.inference_mode()
    def embed_image(self, image_path: str):

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_image_features(**inputs)

            embedding = F.normalize(outputs, p=2, dim=1)

            return embedding.cpu().numpy()[0].tolist()

        except Exception as e:
            logger.error(f"❌ Image embed error: {e}")
            return None

    # =========================
    # TEXT EMBEDDING (FIX 77 TOKENS)
    # =========================
    @torch.inference_mode()
    def embed_text(self, text: str):
        text = text[:200]   # hoặc tokenizer-basedf
        
        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                truncation=True,     # 🔥 FIX crash
                padding=True,
                max_length=77        # 🔥 CLIP limit
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_text_features(**inputs)

            embedding = F.normalize(outputs, p=2, dim=1)

            return embedding.cpu().numpy()[0]

        except Exception as e:
            logger.error(f"❌ Text embed error: {e}")
            return None