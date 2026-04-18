from typing import Dict, Any
from core.schema.base_schema import BaseSchema


class LifestyleSchema:

    @staticmethod
    def transform(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:

        payload = {
            "type": "lifestyle",
            "dataset": dataset_name,

            "user": {
                "age": row.get("Age"),
                "gender": row.get("Gender")
            },

            "health": {
                "bmi": row.get("BMI"),
                "calories": row.get("Calories"),
                "water_intake": row.get("Water_Intake (liters)")
            },

            "activity": {
                "frequency": row.get("Workout_Frequency (days/week)"),
                "type": row.get("Workout_Type")
            },

            "diet": {
                "carbs": row.get("Carbs"),
                "protein": row.get("Proteins"),
                "fat": row.get("Fats")
            },

            "raw": row
        }

        payload["text"] = BaseSchema.build_text(payload)
        return payload