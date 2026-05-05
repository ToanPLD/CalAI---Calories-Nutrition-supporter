# api/routes/food_analysis.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
import io

from core.pipelines.food_analysis_pipeline import FoodAnalysisPipeline

router = APIRouter(prefix="/api/food", tags=["Food Analysis"])

_pipeline = None


def get_food_analysis_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = FoodAnalysisPipeline()
    return _pipeline


@router.post("/analyze")
async def analyze_food_image(
    file: UploadFile = File(...),
    question: str = Form(default="")
):

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image.thumbnail((512, 512))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        result = await get_food_analysis_pipeline().analyze(
            image=image,
            filename=file.filename,
            question=question
        )

    except Exception as e:
        print("[FoodAnalysis] Pipeline error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {type(e).__name__}")

    return JSONResponse(content=result)
