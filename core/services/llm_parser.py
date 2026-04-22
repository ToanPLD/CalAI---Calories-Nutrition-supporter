import json
import requests


class LLMParser:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"

    def parse(self, query: str):

        prompt = f"""
Bạn là AI phân tích dữ liệu.

Hãy trích xuất intent từ câu hỏi sau và trả về JSON:

Schema:
{{
  "metric": "calories | protein | fat | carb",
  "operation": "top | compare | average | distribution",
  "filters": {{
      "calories_lt": number,
      "protein_gt": number
  }},
  "group_by": "category | food_name | none",
  "chart": "bar | pie | line"
}}

Chỉ trả JSON. Không giải thích.

Câu hỏi: {query}
"""

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        text = response.json()["response"]

        try:
            return json.loads(text)
        except:
            print("❌ LLM parse lỗi:", text)
            return {
                "metric": "calories",
                "operation": "top",
                "filters": {},
                "group_by": "food_name",
                "chart": "bar"
            }