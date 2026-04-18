def to_float(x):
    try:
        return float(x)
    except:
        return 0.0


def normalize_macros(data: dict):
    return {
        "calories": to_float(data.get("calories")),
        "protein": to_float(data.get("protein")),
        "carbs": to_float(data.get("carbs")),
        "fat": to_float(data.get("fat")),
    }