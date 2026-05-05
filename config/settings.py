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
        "food_recipes_vectors_768",
        "meta_vectors"
    ]

    FOOD_RAG_COLLECTIONS: List[str] = [
        "food_recipes_vectors_768",
        "recipes_vectors_768",
        "food_text_vectors_768",
        "food_common_vectors_768",
        "food_global_10k_vectors_768",
        "food_nutrition_vectors_768",
        "food_nutrition_dev_vectors_768",
        "food_vectors_768",
        "food_fruit_vectors_768"
    ]

    RECIPE_IMAGE_DATASET: str = (
        "pes12017000148/food-ingredients-and-recipe-dataset-with-images"
    )
    RECIPE_IMAGE_DATASET_COLLECTION: str = "food_ingredients_recipes_multimodal"
    RECIPE_IMAGE_META_DOMAIN: str = "recipe_image"

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
    REDIS_CACHE_COMPRESS_MIN_BYTES: int = 2048
    REDIS_CACHE_MAX_VALUE_BYTES: int = 250000
    AGENTIC_CACHE_TTL: int = 1800
    AGENTIC_CACHE_ENABLED: bool = False

    VISION_MODEL: str = "llava:7b"
    VISION_API_URL: str = "http://localhost:11434/api/generate"
    VISION_TIMEOUT_SECONDS: float = 90.0
    VISION_IMAGE_MAX_SIDE: int = 640
    VISION_IMAGE_JPEG_QUALITY: int = 60
    IMAGE_CLASSIFIER_ENABLED: bool = True
    IMAGE_CLASSIFIER_BACKBONE: str = "clip-vit+resnet50-cnn"
    IMAGE_CLASSIFIER_MODEL: str = "openai/clip-vit-base-patch32"
    IMAGE_CLASSIFIER_CNN_ENABLED: bool = True
    IMAGE_CLASSIFIER_CNN_MODEL: str = "torchvision/resnet50-imagenet1k"
    IMAGE_CLASSIFIER_TOP_K: int = 5
    IMAGE_CLASSIFIER_MIN_CONFIDENCE: float = 0.08

    LLM_MODEL: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest"
    LLM_API_URL: str = "http://localhost:11434/api/generate"
    LLM_BACKEND: str = "ollama"  # ollama | openai
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_NUM_PREDICT: int = 512


settings = Settings()
