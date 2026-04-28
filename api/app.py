from fastapi import FastAPI
from core.agent.agent import DataAgent

app = FastAPI()
agent = DataAgent()


@app.get("/query")
def query(q: str):

    result = agent.run(q)

    if result["type"] == "plan":
        return {
            "type": "plan",
            "content": result["plan"]
        }

    df = result["data"]
    chart_path = result["chart"]

    data = df.to_dict(orient="records") if df is not None else []

    return {
        "type": "chart" if chart_path else "text",
        "chart_path": f"http://localhost:8000/{chart_path}" if chart_path else None,
        "data": data,
        "explanation": result["explanation"]
    }
