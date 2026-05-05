import httpx
import json
import re
from typing import Any

from config.settings import settings
from core.prompts.agentic_prompts import (
    AGENTIC_SYSTEM_PROMPT,
    build_agentic_answer_prompt,
    build_food_image_answer_prompt,
)


class LLMService:

    def __init__(self):
        self.url = settings.LLM_API_URL
        self.model = settings.LLM_MODEL
        self.backend = settings.LLM_BACKEND.lower().strip()
        self.timeout = settings.LLM_TIMEOUT_SECONDS

    # =========================
    # COMMON CALL
    # =========================
    async def _call_llm(self, prompt, temperature=0.3, num_predict=None):
        if self.backend == "openai":
            return await self._call_openai_compatible(
                prompt=prompt,
                temperature=temperature,
                max_tokens=num_predict or settings.LLM_NUM_PREDICT
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": AGENTIC_SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.url, json=payload)
                res.raise_for_status()

            data = res.json()

            text = data.get("response") or data.get("message", {}).get("content")

            if not text:
                return {"error": "No response", "raw": data}

            text = text.strip()

            return self._strip_code_fence(text)

        except Exception as e:
            return {"error": str(e)}

    async def _call_openai_compatible(self, prompt, temperature=0.3, max_tokens=650):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": AGENTIC_SYSTEM_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.url, json=payload)
                res.raise_for_status()

            data = res.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not text:
                return {"error": "No response", "raw": data}
            return self._strip_code_fence(text.strip())

        except Exception as e:
            return {"error": str(e)}

    def _strip_code_fence(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text)
            text = text.replace("```", "")
        return text.strip()

    def _extract_json_object(self, text: str):
        text = self._strip_code_fence(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        return json.loads(match.group())

    def _safe_value(self, value: Any):
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 240:
                return value[:237] + "..."
            return value

        if isinstance(value, dict):
            return {
                str(k): self._safe_value(v)
                for k, v in value.items()
                if k not in {"vector", "embedding", "image", "image_path"}
            }

        if isinstance(value, (list, tuple)):
            return [self._safe_value(v) for v in value[:8]]

        return str(value)[:240]

    def _compact_context(self, context, limit=8):
        if not context:
            return []

        compacted = []
        for item in context[:limit]:
            payload = getattr(item, "payload", item)
            if payload is None:
                continue
            compacted.append(self._safe_value(payload))

        return compacted

    def _first_present(self, payload, keys):
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return key, self._safe_value(value)
        return None, None

    def _compact_agentic_context(self, context, limit=6):
        if not context:
            return []

        name_keys = [
            "title", "Name", "Shrt_Desc", "recipe_name", "name", "dish_name",
            "food", "food_name", "product_name", "Activity, Exercise or Sport (1 hour)",
            "Activity", "Subtype"
        ]
        nutrient_keys = [
            "serving_size", "GmWt_Desc1", "GmWt_1",
            "calories", "Energ_Kcal", "energy-kcal_100g", "Caloric Value",
            "protein", "Protein", "Protein_(g)", "proteins_100g",
            "carbohydrate", "carbs", "Carbohydrates", "Carbohydrt_(g)", "carbohydrates_100g",
            "fat", "Fat", "total_fat", "Lipid_Tot_(g)", "fat_100g",
            "fiber", "Fiber_TD_(g)", "fiber_100g",
            "sodium", "Sodium_(mg)", "sodium_mg"
        ]
        exercise_keys = [
            "Duration (min)", "Distance (km)", "METs", "Calories per kg",
            "130 lb", "155 lb", "180 lb", "205 lb",
            "50 kg (110 lb)", "60 kg (132 lb)", "70 kg (154 lb)",
            "80 kg (176 lb)", "90 kg (198 lb)", "100 kg (220 lb)"
        ]
        text_keys = [
            "ingredients", "Ingredients", "cleaned_ingredients_list",
            "instructions", "Directions", "directions", "description"
        ]

        compacted = []
        for raw_item in context[:limit]:
            payload = getattr(raw_item, "payload", raw_item) or {}
            if not isinstance(payload, dict):
                compacted.append(self._safe_value(payload))
                continue

            item = {}
            name_key, name_value = self._first_present(payload, name_keys)
            if name_key:
                item["name"] = name_value

            for key in ["domain", "source_collection", "source_dataset", "source_row"]:
                if payload.get(key) not in (None, ""):
                    item[key] = self._safe_value(payload.get(key))

            for key in nutrient_keys + exercise_keys + text_keys:
                value = payload.get(key)
                if value not in (None, ""):
                    item[key] = self._safe_value(value)

            if len(item) <= 3:
                for key, value in payload.items():
                    if key in item or key in {"vector", "embedding", "image", "image_path"}:
                        continue
                    if value in (None, "") or isinstance(value, (dict, list, tuple)):
                        continue
                    item[key] = self._safe_value(value)
                    if len(item) >= 10:
                        break

            compacted.append(item)

        return compacted

    def _context_lines(self, context):
        lines = []
        for index, item in enumerate(context or [], start=1):
            if not isinstance(item, dict):
                lines.append(f"{index}. {item}")
                continue

            name = (
                item.get("name")
                or item.get("food")
                or item.get("title")
                or item.get("Activity")
                or item.get("Activity, Exercise or Sport (1 hour)")
                or "item"
            )
            metrics = []
            for label, keys in [
                ("kcal", ["calories", "Caloric Value", "Energ_Kcal", "energy-kcal_100g"]),
                ("protein", ["protein", "Protein", "Protein_(g)", "proteins_100g"]),
                ("carb", ["carbohydrate", "carbs", "Carbohydrates", "Carbohydrt_(g)", "carbohydrates_100g"]),
                ("fat", ["fat", "Fat", "total_fat", "Lipid_Tot_(g)", "fat_100g"]),
                ("serving", ["serving_size", "GmWt_Desc1"]),
            ]:
                for key in keys:
                    value = item.get(key)
                    if value not in (None, ""):
                        metrics.append(f"{label}={value}")
                        break
            lines.append(f"{index}. {name}: {', '.join(metrics) if metrics else 'no numeric metrics'}")
        return "\n".join(lines)

    def _first_metric(self, item, keys):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return "—"

    def _context_metric_table(self, context):
        rows = ["name | serving | kcal | protein | carb | fat"]
        for item in context or []:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("name")
                or item.get("food")
                or item.get("title")
                or item.get("Activity")
                or item.get("Activity, Exercise or Sport (1 hour)")
                or "item"
            )
            rows.append(
                " | ".join(str(value) for value in [
                    name,
                    self._first_metric(item, ["serving_size", "GmWt_Desc1"]),
                    self._first_metric(item, ["calories", "Caloric Value", "Energ_Kcal", "energy-kcal_100g"]),
                    self._first_metric(item, ["protein", "Protein", "Protein_(g)", "proteins_100g"]),
                    self._first_metric(item, ["carbohydrate", "carbs", "Carbohydrates", "Carbohydrt_(g)", "carbohydrates_100g"]),
                    self._first_metric(item, ["fat", "Fat", "total_fat", "Lipid_Tot_(g)", "fat_100g"]),
                ])
            )
        return "\n".join(rows)

    def _has_nutrition_values(self, nutrition):
        if not isinstance(nutrition, dict):
            return False
        for key in ["calories", "kcal", "protein", "carbs", "carbohydrate", "fat"]:
            value = nutrition.get(key)
            if value not in (None, "", 0):
                return True
        return False

    def _food_image_lacks_nutrition(self, analysis):
        if not isinstance(analysis, dict):
            return True
        summary = analysis.get("nutrition_summary") or {}
        return (
            analysis.get("nutrition_source") == "not_available"
            and not self._has_nutrition_values(analysis.get("estimated_nutrition"))
            and not self._has_nutrition_values(summary.get("estimated_visible_portion"))
        )

    def _text_has_unsupported_nutrition_numbers(self, text):
        normalized = str(text or "").lower()
        if re.search(r"[\u4e00-\u9fff]", normalized):
            return True
        if not re.search(r"\d", normalized):
            return False
        nutrition_terms = [
            "kcal", "calo", "calorie", "calories", "protein", "đạm",
            "carb", "carbs", "carbonhydrate", "carbohydrate", "fat",
            "chất béo", "gram", "grams", "g ", "卡路里", "蛋白",
        ]
        return any(term in normalized for term in nutrition_terms)

    def _grounded_food_image_answer(self, question, analysis):
        analysis = analysis if isinstance(analysis, dict) else {}
        vision = analysis.get("vision_detail") or {}
        dish = analysis.get("dish_name") or "món trong ảnh"
        confidence = analysis.get("confidence")
        confidence_text = ""
        if isinstance(confidence, (int, float)):
            confidence_text = f" (độ tin cậy khoảng {round(float(confidence) * 100)}%)"

        ingredients = vision.get("ingredients") or []
        ingredient_text = ""
        if ingredients:
            ingredient_text = " Mình thấy/có thể suy luận các thành phần chính: " + ", ".join(
                str(item) for item in ingredients[:4]
            ) + "."

        if self._food_image_lacks_nutrition(analysis):
            return (
                f"Khả năng cao đây là {dish}{confidence_text}.{ingredient_text} "
                "Mình chưa có đủ dữ liệu khẩu phần từ ảnh để ước tính calories và macro đáng tin cậy. "
                "Bạn cho mình biết khẩu phần khoảng bao nhiêu bát/gram hoặc thành phần chính nhé?"
            ).strip()

        nutrition = analysis.get("estimated_nutrition") or (
            (analysis.get("nutrition_summary") or {}).get("estimated_visible_portion") or {}
        )
        metrics = []
        for label, key, unit in [
            ("calories", "calories", "kcal"),
            ("protein", "protein", "g"),
            ("carb", "carbs", "g"),
            ("fat", "fat", "g"),
        ]:
            value = nutrition.get(key)
            if value not in (None, "", 0):
                metrics.append(f"{label}: {value} {unit}")
        metric_text = "; ".join(metrics)
        return (
            f"Khả năng cao đây là {dish}{confidence_text}.{ingredient_text} "
            f"Ước tính cho phần nhìn thấy: {metric_text}."
        ).strip()

    def _is_low_value_answer(self, text):
        normalized = str(text or "").strip()
        if len(normalized) < 45:
            return True
        table_tokens = normalized.replace(" ", "").lower()
        return table_tokens in {
            "món|khẩuphần|kcal|p|c|f",
            "|món|khẩuphần|kcal|p|c|f|",
        }

    async def _retry_agentic_short(self, query, intent, compact_context, citations):
        prompt = f"""
Cau hoi: {query}
Intent: {intent}
Bang du lieu duoc phep dung, khong duoc sua so:
{self._context_metric_table(compact_context)}

Hay tra loi bang tieng Viet tu nhien.
Neu lap thuc don: chuyen tung dong trong bang du lieu thanh Markdown table voi cot Mon | Khau phan | kcal | Protein | Carb | Fat.
Chi copy so lieu tu bang du lieu. Khong tinh lai. Khong them mon moi. Khong nhac 7700 tru khi cau hoi ve tang/giam can.
Nguon: {json.dumps((citations or [])[:3], ensure_ascii=False, separators=(",", ":"))}
""".strip()
        text = await self._call_llm(
            prompt,
            temperature=0.2,
            num_predict=min(260, max(settings.LLM_NUM_PREDICT, 220))
        )
        if isinstance(text, dict):
            return None
        return text

    async def answer_food_image(self, question, analysis):
        prompt = build_food_image_answer_prompt(question=question, analysis=analysis)
        text = await self._call_llm(
            prompt,
            temperature=0.2,
            num_predict=settings.LLM_NUM_PREDICT
        )
        if isinstance(text, dict):
            return None
        return text

    async def answer_agentic(
        self,
        query,
        intent,
        context,
        citations,
        conversation_context=None,
        user_profile_text=None
    ):
        compact_context = self._compact_agentic_context(context, limit=4)
        prompt = build_agentic_answer_prompt(
            query=query,
            intent=intent,
            context=compact_context,
            citations=citations,
            conversation_context=conversation_context,
            user_profile_text=user_profile_text
        )
        text = await self._call_llm(
            prompt,
            temperature=0.25,
            num_predict=settings.LLM_NUM_PREDICT
        )
        if isinstance(text, dict):
            return None
        if self._is_low_value_answer(text):
            retry_text = await self._retry_agentic_short(
                query=query,
                intent=intent,
                compact_context=compact_context,
                citations=citations
            )
            if retry_text and not self._is_low_value_answer(retry_text):
                return retry_text
        return text

    # =========================
    # TEXT → QA
    # =========================
    async def answer_question(self, question, context):
        compact_context = self._compact_context(context, limit=10)

        prompt = f"""
CÂU HỎI:
{question}

DỮ LIỆU TRUY XUẤT:
{json.dumps(compact_context, ensure_ascii=False)}

Trả lời:
"""

        text = await self._call_llm(
            prompt,
            temperature=0.25,
            num_predict=settings.LLM_NUM_PREDICT
        )

        if isinstance(text, dict):
            return {
                "question": question,
                "answer": "Không thể tạo câu trả lời vì LLM đang lỗi hoặc không phản hồi.",
                "error": text,
                "context_used": compact_context[:5],
                "format": "messenger_text"
            }

        return {
            "question": question,
            "answer": text,
            "context_used": compact_context[:5],
            "format": "messenger_text"
        }
