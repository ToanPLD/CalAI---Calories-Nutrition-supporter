class UnitNormalizer:

    @staticmethod
    def clean(row):
        data = {}

        for k, v in row.items():
            if v is None or str(v).strip() == "":
                data[k] = None
            else:
                data[k] = v

        return data

    # ================= FOOD =================
    @staticmethod
    def normalize_food(row):
        data = UnitNormalizer.clean(row.to_dict())

        # normalize calories key
        if "Caloric Value" in data:
            data["calories"] = data["Caloric Value"]

        return data

    # ================= BEVERAGE =================
    @staticmethod
    def normalize_beverage(row):
        data = UnitNormalizer.clean(row.to_dict())

        if "Calories" in data:
            data["calories"] = data["Calories"]

        return data

    # ================= EXERCISE =================
    @staticmethod
    def normalize_exercise(row):
        data = UnitNormalizer.clean(row.to_dict())

        if "METs" in data:
            data["mets"] = data["METs"]

        return data

    # ================= LIFESTYLE =================
    @staticmethod
    def normalize_lifestyle(row):
        return UnitNormalizer.clean(row.to_dict())