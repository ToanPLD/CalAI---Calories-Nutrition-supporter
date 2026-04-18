from typing import Dict, Any
from core.schema.base_schema import BaseSchema


class BeverageSchema:

    @staticmethod
    def transform(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:

        payload = {
            "type": "beverage",
            "dataset": dataset_name,

            "name": row.get("drink") or row.get("Beverage"),
            "category": row.get("type") or row.get("Beverage_category"),

            "nutrition": {
                "calories": row.get("Calories"),
                "sugar_g": row.get("sugars(g)"),
                "fat_g": row.get("Total Fat (g)"),
                "carbs_g": row.get("Total Carbohydrates (g)")
            },

            "caffeine_mg": row.get("Caffeine (mg)"),
            "volume_ml": row.get("Volume (ml)"),

            "raw": row
        }

        payload["text"] = BaseSchema.build_text(payload)
        return payload