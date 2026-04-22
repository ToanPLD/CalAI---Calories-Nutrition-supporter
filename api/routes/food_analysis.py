from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from core.pipelines.food_analysis_pipeline import FoodAnalysisPipeline

router = APIRouter(prefix="/api/food", tags=["Food Analysis"])

@router.post("/analyze")
async def analyze_food_image(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(400, "Invalid file type")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1024, 1024))
    except Exception as e:
        raise HTTPException(400, str(e))

    pipeline = FoodAnalysisPipeline()
    result = await pipeline.analyze(image=image)

    return JSONResponse(content=result)