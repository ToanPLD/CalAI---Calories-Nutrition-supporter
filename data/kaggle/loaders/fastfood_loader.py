from .base_loader import BaseCSVLoader


class FastFoodLoader(BaseCSVLoader):

    def parse(self, row):
        return {
            "food_name": row.get("Item"),
            "calories": row.get("Calories"),
            "protein": row.get("Protein(g)"),
            "carbs": row.get("Carbs(g)"),
            "fat": row.get("Total Fat(g)"),
            "sodium": row.get("Sodium (mg)"),
            "category": "fast_food",
            "company": row.get("Company"),
        }