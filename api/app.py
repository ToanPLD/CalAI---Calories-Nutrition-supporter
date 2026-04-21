import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from core.agent.agent import DataAgent

app = FastAPI()
agent = DataAgent()

# ================= STATIC =================
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
app.mount("/images", StaticFiles(directory="data/storage/images"), name="images")


# ================= QUERY =================
@app.get("/query")
def query(q: str):

    result = agent.run(q)

    # 🔥 SAFE UNPACK
    if isinstance(result, dict):
        df = result.get("df")
        chart_path = result.get("chart")
        plan = result.get("plan")
    else:
        # fallback
        df = result[0]
        chart_path = result[1] if len(result) > 1 else None
        plan = None

    if df is None or df.empty:
        return {
            "type": "text",
            "chart_path": None,
            "data": [],
            "plan": plan
        }

    data = df.to_dict(orient="records")

    # ================= FIX IMAGE =================
    for item in data:
        path = item.get("image_path")

    # 🔥 FIX: check kiểu dữ liệu
        if isinstance(path, str) and path.strip():
            filename = os.path.basename(path)
            item["image_url"] = f"http://localhost:8000/images/{filename}"
        else:
            item["image_url"] = None

    # ================= FIX CHART =================
    chart_url = None
    if chart_path:
        chart_url = f"http://localhost:8000/{chart_path}"

    return {
        "type": "chart" if chart_url else "text",
        "chart_path": chart_url,
        "data": data,
        "plan": plan
    }