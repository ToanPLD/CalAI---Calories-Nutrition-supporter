# api/routes/food_analysis.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
import io

from core.pipelines.food_analysis_pipeline import FoodAnalysisPipeline

router = APIRouter(prefix="/api/food", tags=["Food Analysis"])

pipeline = FoodAnalysisPipeline()


@router.post("/analyze")
async def analyze_food_image(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 2 * 1024 * 1024: #2mb
            raise HTTPException(status_code=400, detail="File too large")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image.thumbnail((512, 512))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        result = await pipeline.analyze(
            image=image,
            filename=file.filename
        )

    except Exception as e:
        print("❌ Pipeline error:", e)
        raise HTTPException(status_code=500, detail="Processing failed")

    return JSONResponse(content=result)
