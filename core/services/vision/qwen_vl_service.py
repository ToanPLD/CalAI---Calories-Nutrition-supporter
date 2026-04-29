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
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=75, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    def _candidate_models(self):
        models = [self.model, *settings.VISION_FALLBACK_MODELS]
        unique = []
        for model in models:
            model = str(model or "").strip()
            if model and model not in unique:
                unique.append(model)
        return unique[:max(1, settings.VISION_MAX_MODEL_ATTEMPTS)]

    async def _post_vision(self, payload, timeout=None):
        timeout = timeout or settings.VISION_TIMEOUT_SECONDS
        last_error = None
        for model in self._candidate_models():
            payload = dict(payload)
            payload["model"] = model
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    res = await client.post(self.url, json=payload)
                data = res.json()
                if data.get("error"):
                    last_error = data.get("error")
                    continue
                data["_model_used"] = model
                return data
            except Exception as exc:
                last_error = str(exc)
                continue
        return {
            "error": last_error or "No vision model returned a usable response",
            "_model_used": None
        }

    async def analyze_food(self, image: Image.Image, filename_hint=None):

        base64_image = self.image_to_base64(image)

        prompt = """
You are CalAI Vision Pro: a senior food-image analyst, clinical nutrition doctor-style advisor, registered-dietitian style estimator, and practical long-term health consultant. Analyze the IMAGE first. The filename is only a weak hint and must never override visible evidence. Do not diagnose disease, prescribe treatment, or claim certainty beyond the image. All nutrition values are estimates for the visible edible portion.

Internal protocol to follow before answering:
1. Image quality audit: clarity, angle, lighting, occlusion, crop, scale references, and how they affect confidence.
2. Visual evidence extraction: list only what is visible, then separate what is inferred from cultural dish patterns.
3. Dish identification: name the most likely dish, include 2-4 alternatives when ambiguous, and justify each with evidence.
4. Portion reasoning: estimate serving count, grams, volume, pieces, bowl/plate size, broth/sauce amount, meat thickness, rice/noodle volume, and uncertainty.
5. Nutrition analysis: estimate calories, protein, carbs, fat, fiber, sugar, sodium, energy density, macro balance, and main drivers of the estimate.
6. Clinical nutrition assessment: discuss strengths, concerns, blood-sugar load, sodium risk, saturated-fat/fried-food risk, protein adequacy, vegetable/fiber adequacy, hydration/broth/sauce caveats, and who should be cautious.
7. Counseling: give practical adjustments for weight loss, muscle gain, blood sugar control, heart-health style eating, and general balanced eating.
8. Table readiness: include compact rows that the UI can render as tables when the user asks to compare, list, plan, or review numbers.
9. Uncertainty: state what cannot be seen, what could change the estimate, and the best follow-up questions.

Output requirements:
- Output ONLY valid JSON. No markdown. No prose outside JSON.
- Write all text values in Vietnamese.
- Use null for unknown numeric values.
- Numeric fields must be plain numbers with no units. Put units in text fields only.
- Keep arrays concise but information-rich. Avoid generic advice that is not tied to the image.
- Do not invent hidden ingredients; mark them as inferred when not visible.
- confidence and probabilities must be 0-1.

Required JSON:
{
  "image_quality": {"clarity": "good | fair | poor", "lighting": "good | fair | poor", "angle": "...", "occlusion": "...", "confidence_impact": "..."},
  "dish_name": "most likely dish name or unknown",
  "possible_dishes": [{"name": "...", "probability": 0.0, "why": "..."}],
  "description": "...",
  "image_observations": ["visible evidence only"],
  "visible_vs_inferred": {"visible": ["..."], "inferred": ["..."], "not_visible": ["..."]},
  "identification_evidence": ["why this dish is likely"],
  "ingredients": ["visible or likely ingredient"],
  "category": "...",
  "visual_form": "bowl | plate | rice plate | noodle soup | soup | salad | sandwich | pizza | sushi platter | packaged product | drink | dessert | snack | mixed meal | unknown",
  "portion_description": "...",
  "portion_estimation": {"servings": null, "estimated_grams": null, "volume_or_count": "...", "method": "...", "uncertainty": "low | medium | high"},
  "sub_items": [{"name": "...", "count": 0, "estimated_amount": "...", "visible_ingredients": ["..."]}],
  "nutrition_estimate": {"calories": null, "protein": null, "carbs": null, "fat": null, "fiber": null, "sugar": null, "sodium_mg": null, "basis": "...", "main_calorie_drivers": ["..."]},
  "health_context": {"cooking_method": "...", "sauce_or_condiment": "...", "estimated_servings": "...", "energy_density": "low | moderate | high | unknown", "processing_level": "minimally processed | mixed | processed | unknown", "macro_balance": "..."},
  "dietary_assessment": {"health_score_0_10": null, "strengths": ["..."], "concerns": ["..."], "suitable_for": ["..."], "caution_for": ["..."]},
  "risk_flags": [{"risk": "...", "severity": "low | medium | high", "reason": "..."}],
  "recommendations": {"for_weight_loss": ["..."], "for_muscle_gain": ["..."], "for_blood_sugar": ["..."], "for_heart_health": ["..."], "healthier_adjustments": ["..."]},
  "table_rows": [{"metric": "Calories", "value": null, "unit": "kcal", "note": "visible portion estimate"}],
  "uncertainty": {"level": "low | medium | high", "reasons": ["..."], "needs_user_input": ["..."]},
  "confidence": 0.0
}

Filename hint:
""".strip()

        payload = {
            "prompt": f"{prompt}\n{filename_hint or ''}",
            "images": [base64_image],
            "stream": False
        }

        try:
            data = await self._post_vision(payload)

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
