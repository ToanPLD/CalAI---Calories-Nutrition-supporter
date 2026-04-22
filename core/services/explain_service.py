import requests

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
Bạn là chuyên gia phân tích, đánh giá và tư vấn dinh dưỡng.

Nhiệm vụ:
- Trả lời rõ ràng đầy đủ ý nghĩa 
- Phân tích dữ liệu dinh dưỡng
- Nếu là so sánh → chỉ so sánh các thực phẩm được cung cấp
- Không được tự thêm dữ liệu
- Không được nói "có nhiều loại thực phẩm giống nhau"
- Nếu không có trong cơ sở dữ liệu hãy bảo người dùng miêu tả rõ hơn

User query:
{query}

Dữ liệu:
{data}


Yêu cầu:
- Trả lời bằng tiếng Việt
- Ngắn gọn (3-5 câu)
- Tập trung vào calories, protein, fat, carb
- Đưa ra nhận xét + gợi ý

Ví dụ:
"Chuối có lượng calo cao hơn táo, phù hợp cho người cần tăng năng lượng.
Táo ít calo hơn nên thích hợp cho chế độ giảm cân."

Phải:
- So sánh trực tiếp từng chỉ số
- Nêu rõ ưu/nhược điểm
- Đưa lời khuyên sử dụng

Không được:
- Nói chung chung
- Lặp lại dữ liệu

Trả lời:
"""

        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            }
        )

        return res.json().get("response", "")