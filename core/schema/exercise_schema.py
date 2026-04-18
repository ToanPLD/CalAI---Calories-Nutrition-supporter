from typing import Dict, Any
from core.schema.base_schema import BaseSchema


class ExerciseSchema:

    @staticmethod
    def transform(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:

        payload = {
            "type": "exercise",
            "dataset": dataset_name,

            "name": row.get("Activity") or row.get("Workout_Type"),
            "category": row.get("Subtype") or row.get("Type of Muscle"),

            "metrics": {
                "calories_burned": row.get("Calories_Burned"),
                "duration_min": row.get("Duration (min)"),
                "met": row.get("METs"),
                "bpm_avg": row.get("Avg_BPM"),
                "bpm_max": row.get("Max_BPM")
            },

            "body": {
                "weight": row.get("Weight (kg)"),
                "height": row.get("Height (m)"),
                "bmi": row.get("BMI"),
                "fat_percent": row.get("Fat_Percentage")
            },

            "raw": row
        }

        payload["text"] = BaseSchema.build_text(payload)
        return payload