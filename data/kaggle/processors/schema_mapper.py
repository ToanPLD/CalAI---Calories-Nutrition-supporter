def normalize(item):
    return {
        "food_name": item.get("food_name", "unknown"),

        "macros": {
            "calories": float(item.get("calories") or 0),
            "protein": float(item.get("protein") or 0),
            "carbs": float(item.get("carbs") or 0),
            "fat": float(item.get("fat") or 0),
        },

        "micros": {
            "sodium": item.get("sodium"),
            "cholesterol": item.get("cholesterol"),
            "fiber": item.get("fiber"),
        },

        "context": {
            "category": item.get("category"),
            "meal_type": item.get("meal_type"),
            "company": item.get("company"),
        }
    }