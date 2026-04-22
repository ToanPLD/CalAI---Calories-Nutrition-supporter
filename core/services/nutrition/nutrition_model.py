import os
import joblib


class NutritionRegressionModel:

    def __init__(self, model_path="models/nutrition_reg.pkl"):

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.ready = True
        else:
            print("⚠️ Nutrition model NOT FOUND → fallback mode")
            self.model = None
            self.ready = False

    def predict(self, features: dict):

        if not self.ready:
            return {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "note": "fallback (no model)"
            }

        X = [list(features.values())]

        pred = self.model.predict(X)[0]

        return {
            "calories": pred[0],
            "protein": pred[1],
            "carbs": pred[2],
            "fat": pred[3]
        }