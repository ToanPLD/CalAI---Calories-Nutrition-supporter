from fastapi import FastAPI, UploadFile, File
import shutil
from pipelines.query_pipeline import QueryPipeline

app = FastAPI()
pipeline = QueryPipeline()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = pipeline.run(path)

    return result