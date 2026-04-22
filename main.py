from fastapi import FastAPI
from api.routes.food_analysis import router as food_router

app = FastAPI(
    title="Food AI API",
    version="1.0"
)

# register route
app.include_router(food_router)


@app.get("/")
def root():
    return {"message": "Food AI is running 🚀"}