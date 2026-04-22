import httpx
import base64
from io import BytesIO
from PIL import Image
import json


class QwenVLService:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-vl:7b"

    def image_to_base64(self, image: Image.Image):
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    async def analyze_food(self, image: Image.Image):

        base64_image = self.image_to_base64(image)

        payload = {
            "model": self.model,
            "prompt": "Describe food in JSON with name, ingredients, category",
            "images": [base64_image],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(self.url, json=payload)

        text = res.json()["response"]

        try:
            return json.loads(text)
        except:
            return {"dish_name": "unknown", "confidence": 0}