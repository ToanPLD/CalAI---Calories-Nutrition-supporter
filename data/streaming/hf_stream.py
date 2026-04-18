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

        os.makedirs("data/storage/images", exist_ok=True)

    async def stream(self):

        for example in self.dataset:

            # ===== SAVE IMAGE =====
            image = example.get("image")
            image_path = None

            if image:
                file_id = str(uuid.uuid4())
                image_path = f"data/storage/images/{file_id}.jpg"
                image.save(image_path)

            # ===== BUILD PAYLOAD =====
            payload = {
                "food_name": "unknown",
                "calories": example.get("calories"),
                "protein_g": example.get("protein"),
                "carbs_g": example.get("carb"),
                "fat_g": example.get("fat"),
                "image_path": image_path
            }

            # ===== BUILD TEXT =====
            text = f"""
            Food with {payload['calories']} kcal,
            {payload['protein_g']}g protein,
            {payload['carbs_g']}g carbs,
            {payload['fat_g']}g fat.
            """

            yield StreamItem(
                image_path=image_path,
                text=text,
                payload=payload
            )