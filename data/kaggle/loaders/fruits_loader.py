from .base_loader import BaseCSVLoader


class FruitsLoader(BaseCSVLoader):

    def parse(self, row):
        return {
            "food_name": row.get("Fruit"),
            "calories": row.get("Calories (kcal)"),
            "protein": row.get("Protein (g)"),
            "carbs": row.get("Carbohydrates (g)"),
            "fat": 0,
            "fiber": row.get("Fiber (g)"),
            "category": "fruit",
        }