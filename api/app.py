from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from core.services.query_pipeline import QueryPipeline

app = FastAPI()

pipeline = QueryPipeline()

# serve chart images
app.mount("/charts", StaticFiles(directory="charts"), name="charts")


@app.get("/query")
def query(q: str):
    return pipeline.run(q)