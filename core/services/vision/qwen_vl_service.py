import httpx
import base64
from io import BytesIO
from PIL import Image
import json
import re

from config.settings import settings


class QwenVLService:

    def __init__(self):
        self.url = settings.VISION_API_URL
        self.model = settings.VISION_MODEL.strip()

    def image_to_base64(self, image: Image.Image):
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    async def analyze_food(self, image: Image.Image, filename_hint=None):

        base64_image = self.image_to_base64(image)

        payload = {
            "model": self.model,
            "prompt": f"""
You are a food vision assistant. Analyze the image and return ONLY valid JSON.

Schema:
{{
  "dish_name": "...",
  "description": "...",
  "ingredients": ["..."],
  "category": "...",
  "visual_form": "whole pizza | slice | rice plate | packaged product | bowl | drink | unknown",
  "portion_description": "...",
  "confidence": 0.0
}}

Rules:
- Use "unknown" when the dish cannot be identified.
- The uploaded filename can be a useful hint, but visual evidence is more important.
- Recognize common Vietnamese plates such as cơm tấm sườn bì chả trứng, phở, bún bò, bánh mì, gỏi cuốn, bún thịt nướng.
- If the food is pizza, identify visible toppings such as cheese, sausage, pepperoni, ham, beef, vegetables, BBQ sauce.
- Do not name a packaged/brand product unless the package or brand is visible in the image.
- Confidence should be based on visual certainty: high for clear dish type, lower for unclear toppings.
- Do not include markdown or extra explanation.

Filename hint:
{filename_hint or ""}
""",
            "images": [base64_image],
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                res = await client.post(self.url, json=payload)

            data = res.json()

            # 🔍 DEBUG (giữ lại để check nếu lỗi)
            print("🔍 QWEN RAW:", data)

            # =========================
            # SAFE EXTRACT RESPONSE
            # =========================
            text = data.get("response")

            if not text:
                text = data.get("message", {}).get("content")

            if not text:
                return {
                    "dish_name": "unknown",
                    "error": "No response from model",
                    "raw": data
                }

            # =========================
            # PARSE JSON
            # =========================
            try:
                return self._loads_food_json(text)

            except Exception:
                return {
                    "dish_name": "unknown",
                    "raw_text": text
                }

        except Exception as e:
            return {
                "dish_name": "unknown",
                "error": str(e)
            }

    def _extract_json(self, text: str):
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text)
            text = text.replace("```", "").strip()

        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        object_match = re.search(r"\{.*\}", text, re.DOTALL)

        if array_match and (
            object_match is None or array_match.start() < object_match.start()
        ):
            return array_match.group()

        if object_match:
            return object_match.group()

        return text

    def _loads_food_json(self, text: str):
        parsed = json.loads(self._extract_json(text))

        if isinstance(parsed, list):
            parsed = next(
                (item for item in parsed if isinstance(item, dict)),
                {}
            )

        if not isinstance(parsed, dict):
            return {"dish_name": "unknown", "raw": parsed}

        dish_name = str(parsed.get("dish_name", "")).strip()
        if dish_name.lower() in {"", "unknown", "none", "null"}:
            parsed["dish_name"] = "unknown"

        return parsed
