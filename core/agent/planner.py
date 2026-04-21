import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are a strict JSON planner.

RULES:
- ONLY English or Vietnamese
- NEVER Chinese
- Output ONLY JSON
- No explanation

Example:
{
  "steps": [
    {"tool": "search", "query": "apple"},
    {"tool": "compute", "compute": "compare"},
    {"tool": "chart", "chart": "bar"}
  ]
}
"""

class Planner:

    def plan(self, query):

        prompt = SYSTEM_PROMPT + "\nUser: " + query

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                "model": "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                "temperature": 0,
                "top_p": 0.9
            }
            },
                timeout=60
        )

            data = response.json()

            # 🔥 DEBUG LOG
            print("🧠 OLLAMA RAW:", data)

            # ================= SAFE EXTRACT =================
            text = None

            if "response" in data:
                text = data["response"]

            elif "message" in data:
                text = data["message"].get("content")

            elif "error" in data:
                print("❌ Ollama error:", data["error"])
                return self.fallback(query)

            else:
                print("❌ Unknown Ollama format")
                return self.fallback(query)

            # ================= PARSE JSON =================
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]

                return json.loads(json_str)

            except Exception as e:
                print("❌ JSON parse fail:", text)
                return self.fallback(query)

        except Exception as e:
            print("❌ Planner crash:", e)
            return self.fallback(query)

    def fallback(self, query):
        return {
            "steps": [
                {"tool": "search", "query": query}
            ]
        }