import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"


class PlanService:

    def generate(self, query):

        prompt = f"""
Bạn là chuyên gia dinh dưỡng và huấn luyện viên thể hình cho chatbox Messenger.

Câu hỏi / mục tiêu người dùng:
{query}

Nhiệm vụ:
- Hiểu mục tiêu thật của người dùng: tăng cân, giảm cân, giữ cân, tăng cơ, kiểm soát calories hoặc cải thiện lối sống.
- Tạo kế hoạch ăn uống/luyện tập phù hợp với mục tiêu được nêu.
- Nếu thiếu cân nặng, chiều cao, tuổi, bệnh nền hoặc mức vận động, nêu giả định ngắn gọn.

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Dùng bảng markdown đầy đủ cột/hàng, dễ đọc trong Messenger.
- Bảng tối thiểu có: Ngày/Bữa, Món hoặc hoạt động, Khẩu phần, Calories ước tính, Protein ước tính, Ghi chú.
- Sau bảng có 3-5 gạch đầu dòng giải thích cách áp dụng.
- Nếu số liệu chỉ là ước tính, ghi rõ là ước tính.

Không được:
- Bịa bệnh lý hoặc chỉ định y khoa.
- Dùng code block.
- Lặp lại dữ liệu không cần thiết.

Trả lời:
"""

        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4
                    }
                },
                timeout=60
            )
            plan = res.json().get("response", "")
        except Exception as e:
            plan = f"Không thể tạo kế hoạch từ LLM: {e}"

        return {
            "type": "plan",
            "plan": plan
        }
