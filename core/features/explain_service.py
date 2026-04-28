import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"


class ExplainService:

    def explain(self, df, query):

        if df is None or df.empty:
            return "Không tìm thấy dữ liệu phù hợp."

        q = query.lower()

        is_compare = any(k in q for k in ["so sánh", "compare", "vs"])

        if is_compare:
            df = df.head(2)
        else:
            df = df.head(15)

        data = df.to_dict(orient="records")

        prompt = f"""
Bạn là chuyên gia phân tích, đánh giá và tư vấn dinh dưỡng cho chatbox Messenger.

Nguyên tắc:
- Trả lời bằng tiếng Việt, rõ ràng, thực tế.
- Chỉ dùng dữ liệu được cung cấp; thiếu dữ liệu thì ghi "-".
- Nếu câu hỏi là so sánh, xếp hạng, danh sách nhiều món hoặc có từ 3 dòng dữ liệu: tạo bảng markdown đầy đủ cột/hàng.
- Cột bảng ưu tiên: Món, Calories, Protein, Carb, Fat, Nhận xét, Phù hợp.
- Sau bảng thêm 2-4 gạch đầu dòng kết luận.
- Không nói chung chung, không bịa số, không dùng code block.

Câu hỏi:
{query}

Dữ liệu:
{json.dumps(data, ensure_ascii=False)}

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
                        "temperature": 0.3
                    }
                },
                timeout=60
            )
            return res.json().get("response", "")
        except Exception as e:
            return f"Không thể tạo giải thích từ LLM: {e}"
