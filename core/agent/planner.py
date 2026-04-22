import requests
import json
import re
from core.agent.schema import ALLOWED_TOOLS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"


SYSTEM_PROMPT = f"""
You are a STRICT JSON planner.

RULES:
- ONLY use these tools: {ALLOWED_TOOLS}
- DO NOT invent new tools
- DO NOT output anything except JSON
- NO markdown
- NO explanation
- ONLY English or Vietnamese

OUTPUT FORMAT:

{{
  "steps": [
    {{"tool": "search", "query": "..."}},
    {{"tool": "compute", "compute": "..."}},
    {{"tool": "chart", "chart": "bar"}}
  ]
}}
"""


def extract_json(text):
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None


class Planner:

    def plan(self, query):

        prompt = SYSTEM_PROMPT + "\nUser: " + query

        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 200
                    }
                },
                timeout=60
            )

            data = res.json()
            print("🧠 RAW:", data)

            text = data.get("response", "")

            plan = extract_json(text)

            if not plan:
                return self.fallback(query)

            clean_steps = []
            for step in plan.get("steps", []):
                if step.get("tool") in ALLOWED_TOOLS:
                    clean_steps.append(step)
                else:
                    print("❌ remove tool:", step)

            if not clean_steps:
                return self.fallback(query)

            return {"steps": clean_steps}

        except Exception as e:
            print("❌ Planner crash:", e)
            return self.fallback(query)

    def fallback(self, query):
        return {
            "steps": [
                {"tool": "search", "query": query},
                {"tool": "compute", "compute": "top"},
                {"tool": "chart", "chart": "bar"}
            ]
        }