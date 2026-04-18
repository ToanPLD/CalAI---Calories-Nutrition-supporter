from .base_loader import BaseCSVLoader


class DailyLoader(BaseCSVLoader):

    def parse(self, row):
        return {
            "food_name": row.get("Food_Item"),
            "calories": row.get("Calories (kcal)"),
            "protein": row.get("Protein (g)"),
            "carbs": row.get("Carbohydrates (g)"),
            "fat": row.get("Fat (g)"),
            "meal_type": row.get("Meal_Type"),
            "category": row.get("Category"),
        }