class UnitNormalizer:

    # =========================
    # CLEAN DATA
    # =========================
    @staticmethod
    def clean(row):
        data = {}

        for k, v in row.items():
            if v is None or str(v).strip() == "":
                data[k] = None
            else:
                data[k] = v

        return data

    # =========================
    # HELPER: GET FIRST EXISTING
    # =========================
    @staticmethod
    def pick(data, keys):
        for k in keys:
            if k in data and data[k] not in [None, ""]:
                return data[k]
        return None

    # =========================
    # FOOD (🔥 QUAN TRỌNG NHẤT)
    # =========================
    @staticmethod
    def normalize_food(row):
        raw = UnitNormalizer.clean(row.to_dict())

        data = dict(raw)  # giữ FULL data

        # ================= NAME =================
        data["food_name"] = UnitNormalizer.pick(raw, [
            "Food",
            "Food_Item",
            "Shrt_Desc",
            "Beverage",
            "meal_name",
            "name"
        ])

        # ================= CALORIES =================
        data["calories"] = UnitNormalizer.pick(raw, [
            "Calories",
            "Calories (kcal)",
            "Caloric Value",
            "Energ_Kcal"
        ])

        # ================= PROTEIN =================
        data["protein"] = UnitNormalizer.pick(raw, [
            "Protein",
            "Protein (g)"
        ])

        # ================= CARB =================
        data["carb"] = UnitNormalizer.pick(raw, [
            "Carbohydrates",
            "Carbohydrates (g)"
        ])

        # ================= FAT =================
        data["fat"] = UnitNormalizer.pick(raw, [
            "Fat",
            "Fat (g)",
            "Total Fat (g)"
        ])

        # ================= CATEGORY =================
        data["category"] = UnitNormalizer.pick(raw, [
            "Category",
            "Beverage_category",
            "meal_type"
        ])

        return data

    # =========================
    # BEVERAGE
    # =========================
    @staticmethod
    def normalize_beverage(row):
        raw = UnitNormalizer.clean(row.to_dict())

        data = dict(raw)

        data["food_name"] = UnitNormalizer.pick(raw, [
            "drink",
            "Beverage",
            "name"
        ])

        data["calories"] = UnitNormalizer.pick(raw, [
            "Calories"
        ])

        data["caffeine"] = UnitNormalizer.pick(raw, [
            "Caffeine (mg)",
            "Caffeine"
        ])

        data["category"] = UnitNormalizer.pick(raw, [
            "type",
            "Beverage_category"
        ])

        return data

    # =========================
    # EXERCISE
    # =========================
    @staticmethod
    def normalize_exercise(row):
        raw = UnitNormalizer.clean(row.to_dict())

        data = dict(raw)

        data["activity"] = UnitNormalizer.pick(raw, [
            "Activity",
            "Workout_Type",
            "Exercise or Sport (1 hour)"
        ])

        data["calories"] = UnitNormalizer.pick(raw, [
            "Calories_Burned",
            "Calories per lb"
        ])

        data["mets"] = UnitNormalizer.pick(raw, [
            "METs"
        ])

        return data

    # =========================
    # LIFESTYLE
    # =========================
    @staticmethod
    def normalize_lifestyle(row):
        raw = UnitNormalizer.clean(row.to_dict())

        data = dict(raw)

        data["age"] = UnitNormalizer.pick(raw, ["Age"])
        data["weight"] = UnitNormalizer.pick(raw, ["Weight", "Weight (kg)"])
        data["height"] = UnitNormalizer.pick(raw, ["Height", "Height (m)"])

        data["calories"] = UnitNormalizer.pick(raw, [
            "Calories",
            "Calories_Burned"
        ])

        data["protein"] = UnitNormalizer.pick(raw, [
            "Proteins"
        ])

        data["carb"] = UnitNormalizer.pick(raw, [
            "Carbs"
        ])

        data["fat"] = UnitNormalizer.pick(raw, [
            "Fats"
        ])

        return data