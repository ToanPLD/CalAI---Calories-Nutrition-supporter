import requests


class OllamaService:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str):
        try:
            res = requests.post(self.url, json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            })

            data = res.json()

            if "response" in data:
                return data["response"]

            if "message" in data:
                return data["message"]["content"]

            return str(data)

        except Exception as e:
            return f"LLM error: {e}"