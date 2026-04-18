from datasets import load_dataset
from dataclasses import dataclass
import uuid
import os


@dataclass
class StreamItem:
    image_path: str
    text: str
    payload: dict


class HFStreamer:

    def __init__(self):
        self.dataset = load_dataset(
            "pinkieseb/nutrition_dataset",
            split="train",
            streaming=True
        )

        self.image_dir = "data/storage/images"
        os.makedirs(self.image_dir, exist_ok=True)

    async def stream(self):

        for example in self.dataset:

            # ================= IMAGE =================
            image_path = None
            image = example.get("image")

            if image:
                try:
                    file_id = str(uuid.uuid4())
                    image_path = os.path.join(self.image_dir, f"{file_id}.jpg")
                    image.save(image_path)
                except Exception:
                    image_path = None

            # ================= SAFE FIELD MAPPING =================
            food_name = (
                example.get("food_name")
                or example.get("name")
                or example.get("label")
                or example.get("Food")
                or "unknown"
            )

            # 👉 bỏ data rác
            if not food_name or food_name == "unknown":
                continue

            payload = {
                "food_name": str(food_name).lower().strip(),

                "calories": self.safe_float(example.get("calories")),
                "protein_g": self.safe_float(example.get("protein_g")),
                "carbs_g": self.safe_float(example.get("carbs_g")),
                "fat_g": self.safe_float(example.get("fat_g")),

                "image_path": image_path,
                "domain": "food"
            }

            # ================= TEXT =================
            text = self.build_text(payload)

            # 👉 tránh embed text rỗng
            if not text or len(text) < 10:
                continue

            yield StreamItem(
                image_path=image_path,
                text=text,
                payload=payload
            )

    # ================= SAFE FLOAT =================
    def safe_float(self, val):
        try:
            return float(val)
        except:
            return 0.0

    # ================= TEXT BUILDER (UPGRADE) =================
    def build_text(self, p):

        # 👉 enrich semantic
        name = p["food_name"]

        return (
            f"{name}. "
            f"This is a food item called {name}. "
            f"It contains {p['calories']} kilocalories. "
            f"Protein content is {p['protein_g']} grams. "
            f"Carbohydrates are {p['carbs_g']} grams. "
            f"Fat content is {p['fat_g']} grams. "
            f"This food can be used for nutrition analysis and diet planning."
        )