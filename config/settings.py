from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


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

    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"          

    TEXT_VECTOR_DIM: int = 768
    IMAGE_VECTOR_DIM: int = 512

    TEXT_COLLECTIONS: List[str] = [
        "beverage_text_vectors_768",
        "exercise_text_vectors_768",
        "food_text_vectors_768",
        "diet_recommendations_vectors",
        "exercise_vectors_768",
        "food_vectors_768",
        "food_nutrition_vectors_768",
        "food_nutrition_dev_vectors_768",
        "food_fruit_vectors_768",
        "food_global_10k_vectors_768",
        "exercise_gym_vectors_768",
        "lifestyle_vectors_768",
        "food_common_vectors_768",
        "lifestyle_obesity_vectors_768",
        "recipes_vectors_768",
        "beverage_vectors_768",
        "food_recipes_vectors_768"
    ]

    LEGACY_COLLECTIONS: List[str] = [
        "food_vectors",
        "food_text_vectors",
        "beverage_vectors",
        "exercise_vectors",
        "lifestyle_vectors"
    ]

    TOP_K: int = 3
    FINAL_TOP_K: int = 5

    HYBRID_WEIGHT: float = 0.6
    VISION_MIN_CONFIDENCE: float = 0.65
    RAG_LOW_CONFIDENCE_THRESHOLD: float = 0.70
    RAG_REJECT_PACKAGED_ON_GENERIC_IMAGE: bool = True
    RAG_USE_QDRANT_NAME_FILTER: bool = False
    RAG_CANDIDATE_TOP_K: int = 20

    BATCH_SIZE: int = 32
    CACHE_TTL: int = 86400 

    VISION_MODEL: str = "llava:7b"
    VISION_API_URL: str = "http://localhost:11434/api/generate"

    LLM_MODEL: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"
    LLM_API_URL: str = "http://localhost:11434/api/generate"


settings = Settings()
