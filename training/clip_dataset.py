# training/clip_dataset.py

from torch.utils.data import Dataset
from PIL import Image
import json


class FoodCLIPDataset(Dataset):

    def __init__(self, jsonl_path):
        self.samples = [json.loads(l) for l in open(jsonl_path)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image_path"]).convert("RGB")

        return {
            "image": image,
            "text": item["text"]
        }