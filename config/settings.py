from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    QDRANT_URL: str
    QDRANT_API_KEY: str
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    

    qdrant_food_knowledge_collection: str = "food_knowledge"
    qdrant_food_nutrition_collection: str = "food_nutrition"
    qdrant_food_recipes_collection: str = "food_recipes_vectors"
    qdrant_recipes_collection: str = "recipes_vectors"

    qdrant_food_image_collection: str = "food_image_vectors"
    qdrant_diet_recommendations_collection: str = "diet_recommendations_vectors"
    qdrant_food_text_collection: str = "food_text_vectors"
    qdrant_food_collection: str = "food_vectors"

    qdrant_beverage_collection: str = "beverage_vectors"
    qdrant_exercise_collection: str = "exercise_vectors"
    qdrant_lifestyle_collection: str = "lifestyle_vectors"

    vit_model: str = "google/vit-base-patch16-224"

    batch_size: int = 8


settings = Settings()