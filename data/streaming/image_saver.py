from pathlib import Path


class ImageSaver:

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, image_obj, item_id: str):
        try:
            if image_obj is None:
                return None

            path = self.base_dir / f"{item_id}.jpg"

            # 🔥 FIX QUAN TRỌNG: mọi ảnh convert về RGB
            if image_obj.mode != "RGB":
                image_obj = image_obj.convert("RGB")

            image_obj.save(path, "JPEG")

            return str(path)

        except Exception as e:
            print("Save image error:", e)
            return None