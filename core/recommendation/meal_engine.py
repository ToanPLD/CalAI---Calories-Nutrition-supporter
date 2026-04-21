def recommend_meal(results, goal="muscle_gain"):

    meal = []

    for r in results:
        p = r.payload

        protein = p.get("protein", 0)
        calories = p.get("calories", 9999)
        fat = p.get("fat", 9999)

        # 🔥 HARD FILTER (QUAN TRỌNG NHẤT)
        if goal == "muscle_gain":

            if (
                protein >= 20
                and calories <= 300
                and fat <= 10
            ):
                meal.append(r)

        elif goal == "weight_loss":

            if (
                calories <= 200
                and fat <= 8
            ):
                meal.append(r)

    return meal[:5]