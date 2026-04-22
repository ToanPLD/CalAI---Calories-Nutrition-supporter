from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # =========================
    # QDRANT (🔥 FIX CHUẨN)
    # =========================
    QDRANT_URL: str
    QDRANT_API_KEY: str

    # =========================
    # COLLECTIONS
    # =========================
    qdrant_image_collection: str = "food_image_vectors"
    qdrant_text_collection: str = "food_text_vectors"

    qdrant_food_image_collection: str = "food_image_vectors"
    qdrant_food_text_collection: str = "food_text_vectors"

    qdrant_beverage_collection: str = "beverage_vectors"
    qdrant_exercise_collection: str = "exercise_vectors"
    qdrant_lifestyle_collection: str = "lifestyle_vectors"

    # =========================
    # PATHS
    # =========================
    image_dir: str = "data/storage/images"

    # =========================
    # MODEL
    # =========================
    vit_model: str = "google/vit-base-patch16-224"

    # =========================
    # BATCH
    # =========================
    batch_size: int = 8


settings = Settings()