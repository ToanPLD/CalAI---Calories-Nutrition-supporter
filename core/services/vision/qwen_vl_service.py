import httpx
import base64
from io import BytesIO
from PIL import Image
import json
import re

from config.settings import settings
from core.prompts.agentic_prompts import FOOD_VISION_PROMPT


class QwenVLService:

    def __init__(self):
        self.url = settings.VISION_API_URL
        self.model = settings.VISION_MODEL.strip()

    def image_to_base64(self, image: Image.Image):
        image = image.convert("RGB")
        max_side = max(384, int(settings.VISION_IMAGE_MAX_SIDE))
        image.thumbnail((max_side, max_side))
        buf = BytesIO()
        image.save(
            buf,
            format="JPEG",
            quality=max(40, min(85, int(settings.VISION_IMAGE_JPEG_QUALITY))),
            optimize=True
        )
        return base64.b64encode(buf.getvalue()).decode()

    async def _post_vision(self, payload, timeout=None):
        timeout = timeout or settings.VISION_TIMEOUT_SECONDS
        payload = dict(payload)
        payload["model"] = self.model
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(self.url, json=payload)
                res.raise_for_status()
            data = res.json()
            if data.get("error"):
                return {"error": data.get("error"), "_model_used": self.model}
            data["_model_used"] = self.model
            return data
        except Exception as exc:
            return {"error": str(exc), "_model_used": self.model}

    async def analyze_food(self, image: Image.Image, filename_hint=None):

        base64_image = self.image_to_base64(image)

        payload = {
            "prompt": f"{FOOD_VISION_PROMPT}\n\nFilename hint:\n{filename_hint or ''}",
            "images": [base64_image],
            "stream": False
        }

        try:
            data = await self._post_vision(payload)

            if data.get("error"):
                print(f"[Vision] Qwen-VL error: {data.get('error')}")

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

    async def caption_food_image(
        self,
        image: Image.Image,
        filename_hint=None,
        title=None,
        ingredients=None
    ):
        base64_image = self.image_to_base64(image)

        prompt = """
You are a food image captioning assistant for a recipe retrieval dataset.
Inspect the image and return ONLY valid JSON. Do not include markdown.
Use Vietnamese for human-facing text.

JSON schema:
{
  "caption": "one concise visual caption",
  "dish_name": "most likely dish title or unknown",
  "visual_tags": ["short searchable visual tag"],
  "visible_ingredients": ["visible ingredient"],
  "cooking_method": "baked | fried | grilled | boiled | raw | mixed | unknown",
  "confidence": 0.0,
  "uncertainty": "what is unclear from the image"
}

Rules:
- Trust the image first.
- The title and ingredients are hints only.
- Keep caption under 35 words.
- confidence must be 0-1.
""".strip()

        hints = {
            "filename_hint": filename_hint,
            "title_hint": title,
            "ingredient_hints": ingredients
        }

        payload = {
            "prompt": f"{prompt}\nHints:\n{json.dumps(hints, ensure_ascii=False)}",
            "images": [base64_image],
            "stream": False
        }

        try:
            data = await self._post_vision(payload, timeout=min(settings.VISION_TIMEOUT_SECONDS, 20))
            text = data.get("response") or data.get("message", {}).get("content")
            if not text:
                return {
                    "caption": None,
                    "dish_name": "unknown",
                    "visual_tags": [],
                    "visible_ingredients": [],
                    "confidence": 0,
                    "error": "No caption response from model"
                }

            return self._loads_json_object(text)

        except Exception as e:
            return {
                "caption": None,
                "dish_name": "unknown",
                "visual_tags": [],
                "visible_ingredients": [],
                "confidence": 0,
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
        parsed = self._loads_json_object(text)

        dish_name = str(parsed.get("dish_name", "")).strip()
        if dish_name.lower() in {"", "unknown", "none", "null"}:
            parsed["dish_name"] = "unknown"

        return parsed

    def _loads_json_object(self, text: str):
        json_text = self._extract_json(text)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            repaired = re.sub(
                r":\s*(-?\d+(?:\.\d+)?)\s*(kcal|calories|calorie|grams|gram|g|mg)\b",
                r": \1",
                json_text,
                flags=re.IGNORECASE
            )
            parsed = json.loads(repaired)

        if isinstance(parsed, list):
            parsed = next(
                (item for item in parsed if isinstance(item, dict)),
                {}
            )

        if not isinstance(parsed, dict):
            return {"dish_name": "unknown", "raw": parsed}

        return parsed
