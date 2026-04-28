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

    def predict(self, features):

        if not self.ready:
            return {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "note": "fallback (no model)"
            }

        if not features:
            return {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "note": "fallback (empty features)"
            }

        if isinstance(features, dict):
            values = list(features.values())
        else:
            values = list(features)

        try:
            pred = self.model.predict([values])[0]
        except Exception as e:
            return {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "note": f"fallback (prediction failed: {e})"
            }

        return {
            "calories": pred[0],
            "protein": pred[1],
            "carbs": pred[2],
            "fat": pred[3]
        }
