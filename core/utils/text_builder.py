class TextBuilder:

    @staticmethod
    def compress(payload, max_len=20):
        return str(payload)[:max_len]

    @staticmethod
    def build_generic(payload, max_fields=12):
        parts = []
        count = 0

        for k, v in payload.items():
            if count >= max_fields:
                break

            if v is None:
                continue

            k = k.replace("_", " ").lower()
            v = TextBuilder.compress(v)

            parts.append(f"{k}:{v}")
            count += 1

        return " | ".join(parts)

    @staticmethod
    def build_food(payload):
        return TextBuilder.build_generic(payload)

    @staticmethod
    def build_beverage(payload):
        return TextBuilder.build_generic(payload)

    @staticmethod
    def build_exercise(payload):
        return TextBuilder.build_generic(payload)

    @staticmethod
    def build_lifestyle(payload):
        return TextBuilder.build_generic(payload)