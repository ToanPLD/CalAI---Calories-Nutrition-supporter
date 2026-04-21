def score_food(item):

    p = item.payload

    protein = float(p.get("protein", 0) or 0)
    calories = float(p.get("calories", 1) or 1)
    fat = float(p.get("fat", 0) or 0)
    carb = float(p.get("carb", 0) or 0)

    # 🔥 PROTEIN DENSITY (QUAN TRỌNG NHẤT)
    protein_density = protein / max(calories, 1)

    # 🔥 FAT RATIO
    fat_ratio = fat / max(calories, 1)

    # 🔥 CARB CONTROL
    carb_ratio = carb / max(calories, 1)

    # ===== FINAL SCORE =====
    score = (
        protein_density * 100     # ưu tiên cực mạnh
        - fat_ratio * 50
        - carb_ratio * 30
        - calories * 0.02         # phạt mạnh calories
    )

    return score

def rank_by_health(results):

    scored = [(score_food(r), r) for r in results]

    scored.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scored]