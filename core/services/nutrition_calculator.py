class NutritionCalculator:

    def calculate_simple(self, payload):

        return {
            "calories": payload.get("calories"),
            "protein": payload.get("protein"),
            "carbs": payload.get("carbs"),
            "fat": payload.get("fat")
        }