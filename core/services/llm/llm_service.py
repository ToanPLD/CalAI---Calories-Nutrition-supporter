import httpx
import json


class LLMService:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"

    async def generate_final(self, vision, nutrition):

        prompt = f"""
Bạn là chuyên gia dinh dưỡng.

Thông tin món ăn:
{json.dumps(vision, ensure_ascii=False)}

Dinh dưỡng:
{json.dumps(nutrition, ensure_ascii=False)}

Hãy trả về JSON:
{{
  "dish_name": "...",
  "description": "...",
  "nutrition": {{
    "calories": ...,
    "protein": ...,
    "carbs": ...,
    "fat": ...
  }},
  "health_advice": "..."
}}
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(self.url, json=payload)

        text = res.json()["response"]

        try:
            return json.loads(text)
        except:
            return {
                "raw": text,
                "vision": vision,
                "nutrition": nutrition
            }