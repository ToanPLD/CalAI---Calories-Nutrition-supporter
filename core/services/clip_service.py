import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

    def embed_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)[0].detach().cpu().numpy().tolist()

    def embed_text(self, text):
        text = text[:200]
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inputs)[0].detach().cpu().numpy().tolist()
    def embed_image_pil(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().numpy().tolist()
