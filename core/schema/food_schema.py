from typing import Dict, Any
from core.schema.base_schema import BaseSchema


class FoodSchema:

    @staticmethod
    def transform(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:

        payload = {
            "type": "food",
            "dataset": dataset_name,

            # ===== BASIC =====
            "name": row.get("Food") or row.get("Food_Item") or row.get("food_name"),
            "category": row.get("Category"),

            # ===== NUTRITION CORE =====
            "nutrition": {
                "calories": row.get("Calories (kcal)") or row.get("Caloric Value") or row.get("Energ_Kcal"),
                "protein_g": row.get("Protein (g)") or row.get("Protein"),
                "carbs_g": row.get("Carbohydrates (g)") or row.get("Carbs"),
                "fat_g": row.get("Fat (g)") or row.get("Fat"),
                "fiber_g": row.get("Fiber (g)"),
                "sugar_g": row.get("Sugars (g)"),
                "sodium_mg": row.get("Sodium (mg)"),
                "cholesterol_mg": row.get("Cholesterol (mg)")
            },

            # ===== EXTENDED NUTRITION =====
            "vitamins": {
                k: v for k, v in row.items() if "Vitamin" in k or "Vit_" in k
            },

            "minerals": {
                k: v for k, v in row.items() if k in [
                    "Calcium", "Iron", "Magnesium", "Potassium",
                    "Zinc", "Phosphorus", "Selenium"
                ]
            },

            # ===== FULL RAW DATA =====
            "raw": row
        }

        payload["text"] = BaseSchema.build_text(payload)
        return payload