import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"


class PlanService:

    def generate(self, query):

        prompt = f"""
Bạn là chuyên gia dinh dưỡng và huấn luyện viên thể hình.

User:
{query}

Nhiệm vụ:
- Tạo lịch trình ăn uống 7 ngày
- Phù hợp mục tiêu tăng cân

Yêu cầu:
- Mỗi ngày gồm:
  - Bữa sáng
  - Bữa trưa
  - Bữa tối
  - Snack
- Có calorie ước lượng
- Thực tế, dễ áp dụng

Không được:
- Nói chung chung
- Lặp lại dữ liệu

Trả lời bằng tiếng Việt:
"""

        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5
                }
            }
        )

        return {
            "type": "plan",
            "plan": res.json().get("response", "")
        }