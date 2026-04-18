class FilterService:

    def apply(self, items, filters: dict):
        results = []

        for item in items:
            payload = item.payload

            ok = True

            if "min_protein" in filters:
                if payload.get("protein_g", 0) < filters["min_protein"]:
                    ok = False

            if "max_carbs" in filters:
                if payload.get("carbs_g", 999) > filters["max_carbs"]:
                    ok = False

            if "max_fat" in filters:
                if payload.get("fat_g", 999) > filters["max_fat"]:
                    ok = False

            if ok:
                results.append(item)

        return results