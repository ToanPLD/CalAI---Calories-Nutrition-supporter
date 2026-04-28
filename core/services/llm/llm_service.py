import httpx
import json
import re
from typing import Any

from config.settings import settings


class LLMService:

    def __init__(self):
        self.url = settings.LLM_API_URL
        self.model = settings.LLM_MODEL

    # =========================
    # COMMON CALL
    # =========================
    async def _call_llm(self, prompt, temperature=0.3, num_predict=None):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                res = await client.post(self.url, json=payload)
                res.raise_for_status()

            data = res.json()
            print("🔍 LLM RAW:", data)

            text = data.get("response") or data.get("message", {}).get("content")

            if not text:
                return {"error": "No response", "raw": data}

            text = text.strip()

            return self._strip_code_fence(text)

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

    # =========================
    # IMAGE → NUTRITION
    # =========================
    async def generate_final(self, vision, nutrition=None, rag=None):
        rag_context = self._compact_context(rag or [], limit=5)

        prompt = f"""
Bạn là chuyên gia dinh dưỡng và fitness coach. Nhiệm vụ của bạn là hợp nhất kết quả nhận diện ảnh với dữ liệu dinh dưỡng được truy xuất.

Nguyên tắc:
- Ưu tiên dữ liệu truy xuất được khi có chỉ số rõ ràng.
- Nếu thiếu dữ liệu, dùng null thay vì bịa số chính xác.
- Nếu phải ước lượng, ghi rõ trong health_advice là ước lượng.
- Trả lời đúng JSON, không markdown, không giải thích ngoài JSON.

INPUT:
- Vision:
{json.dumps(vision, ensure_ascii=False)}

- Nutrition data:
{json.dumps(nutrition or {}, ensure_ascii=False)}

- Retrieved context:
{json.dumps(rag_context, ensure_ascii=False)}

JSON schema bắt buộc, các chỉ số có thể là number hoặc null:
{{
  "dish_name": "...",
  "description": "...",
  "nutrition": {{
    "calories": 0,
    "protein": 0,
    "carbs": 0,
    "fat": 0
  }},
  "confidence_note": "...",
  "health_advice": "..."
}}
"""

        text = await self._call_llm(prompt, temperature=0.2, num_predict=500)

        if isinstance(text, dict):  # error case
            return text

        try:
            return self._extract_json_object(text)
        except Exception:
            return {
                "error": "JSON parse failed",
                "raw_text": text,
                "vision": vision,
                "nutrition": nutrition
            }

    # =========================
    # TEXT → QA
    # =========================
    async def answer_question(self, question, context):
        compact_context = self._compact_context(context, limit=10)

        prompt = f"""
Bạn là trợ lý RAG chuyên gia về dinh dưỡng, thực phẩm, đồ uống, vận động và lối sống.

Mục tiêu:
- Hiểu sâu ý định thật của người hỏi trước khi trả lời.
- Trả lời bằng tiếng Việt tự nhiên, rõ ràng, có tính tư vấn thực tế.
- Dựa trên DỮ LIỆU được cung cấp. Nếu thiếu dữ liệu, nói rõ phần nào chưa đủ và đưa cách hỏi bổ sung.
- Không bịa chỉ số dinh dưỡng, không biến ước lượng thành sự thật.
- Không dùng code block. Nội dung phải dễ đọc trong chatbox Messenger.

Quy tắc định dạng linh hoạt:
- Nếu câu hỏi là so sánh, xếp hạng, lựa chọn nhiều món, thực đơn, lịch ăn/tập, hoặc có từ 3 mục dữ liệu trở lên: tạo bảng markdown đầy đủ cột và hàng.
- Bảng nên có các cột phù hợp như: Mục, Calories, Protein, Carb, Fat, Điểm mạnh, Lưu ý. Với thực đơn/lịch: Ngày/Bữa, Món, Khẩu phần, Calories, Protein, Ghi chú.
- Nếu dữ liệu thiếu ở ô nào, ghi "-"; không tự chế số.
- Sau bảng, thêm 2-4 gạch đầu dòng kết luận và khuyến nghị.
- Nếu câu hỏi đơn giản: trả lời trực tiếp 3-6 câu, chỉ thêm bảng khi thật sự giúp người dùng hiểu nhanh hơn.
- Nếu nội dung có rủi ro sức khỏe, thêm nhắc nhở ngắn rằng đây không thay thế tư vấn y tế cá nhân.

CÂU HỎI:
{question}

DỮ LIỆU TRUY XUẤT:
{json.dumps(compact_context, ensure_ascii=False)}

Trả lời:
"""

        text = await self._call_llm(prompt, temperature=0.25, num_predict=900)

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
