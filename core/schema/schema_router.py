from core.schema.food_schema import FoodSchema
from core.schema.beverage_schema import BeverageSchema
from core.schema.exercise_schema import ExerciseSchema
from core.schema.lifestyle_schema import LifestyleSchema


class SchemaRouter:

    DATASET_MAP = {
        # FOOD
        "fastfood": FoodSchema,
        "nutrition": FoodSchema,
        "fruits": FoodSchema,
        "daily_food": FoodSchema,

        # BEVERAGE
        "caffeine": BeverageSchema,
        "starbucks": BeverageSchema,

        # EXERCISE
        "gym": ExerciseSchema,
        "calories_burn": ExerciseSchema,

        # LIFESTYLE
        "lifestyle": LifestyleSchema,
        "obesity": LifestyleSchema
    }

    @staticmethod
    def transform(dataset_key, row):
        schema = SchemaRouter.DATASET_MAP.get(dataset_key)

        if not schema:
            raise ValueError(f"Unsupported dataset: {dataset_key}")

        return schema.transform(row, dataset_key)