from .base_loader import BaseCSVLoader


class NutritionLoader(BaseCSVLoader):

    def parse(self, row):
        return {
            "food_name": row.get("name"),
            "calories": row.get("calories"),
            "protein": row.get("protein"),
            "carbs": row.get("carbohydrates"),
            "fat": row.get("total_fat"),
            "sodium": row.get("sodium"),
            "cholesterol": row.get("cholesterol"),
            "category": "generic_food",
        }