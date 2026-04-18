import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService
from core.utils.unit_normalizer import UnitNormalizer
from core.utils.text_builder import TextBuilder
from data.kaggle.dataset_registry import DATASETS


class KaggleIngestionPipeline:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    # =========================
    # AUTO DETECT FILE
    # =========================
    def load_dataset(self, name):

        path = kagglehub.dataset_download(name)

        selected_file = None

        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".csv", ".xlsx", ".json", ".parquet")):
                    selected_file = os.path.join(root, f)
                    print(f"📂 Using file: {selected_file}")
                    break
            if selected_file:
                break

        if not selected_file:
            raise ValueError(f"No valid file found in {name}")

    # ✅ READ DIRECTLY (KHÔNG dùng kagglehub.load_dataset nữa)
        if selected_file.endswith(".csv"):
            return pd.read_csv(selected_file)

        elif selected_file.endswith(".xlsx"):
            return pd.read_excel(selected_file)

        elif selected_file.endswith(".json"):
            return pd.read_json(selected_file)

        elif selected_file.endswith(".parquet"):
            return pd.read_parquet(selected_file)

        else:
            raise ValueError(f"Unsupported file: {selected_file}")

 

    # =========================
    # MAIN PIPELINE
    # =========================
    def run(self):

        for ds in DATASETS:
            print(f"\n🚀 Processing {ds['name']}")

            df = self.load_dataset(ds["name"])  # 🔥 FIX HERE

            batch = []

            for _, row in df.iterrows():

                # ================= NORMALIZE =================
                payload = self.normalize(row, ds["handler"])

                if not payload:
                    continue

                payload["domain"] = ds["domain"]

                # ================= BUILD TEXT =================
                text = self.build_text(payload, ds["domain"])

                if not text:
                    continue

                # ================= EMBED =================
                vector = self.clip.embed_text(text)

                if vector is None:
                    continue

                vector = vector.tolist() if hasattr(vector, "tolist") else vector

                batch.append({
                    "vector": vector,
                    "payload": payload
                })

                # ================= FLUSH =================
                if len(batch) >= 64:
                    self.upsert(ds["domain"], batch)
                    batch.clear()

            # final flush
            if batch:
                self.upsert(ds["domain"], batch)

            print(f"✅ Done {ds['name']}")

    # ================= ROUTER =================

    def normalize(self, row, handler):

        if handler == "food_full":
            return UnitNormalizer.normalize_food(row)

        if handler in ["beverage", "beverage_full"]:
            return UnitNormalizer.normalize_beverage(row)

        if handler in ["exercise", "exercise_table", "exercise_mets"]:
            return UnitNormalizer.normalize_exercise(row)

        if handler in ["lifestyle", "lifestyle_full"]:
            return UnitNormalizer.normalize_lifestyle(row)

        return {}

    def build_text(self, payload, domain):

        if domain == "food":
            return TextBuilder.build_food(payload)

        if domain == "beverage":
            return TextBuilder.build_beverage(payload)

        if domain == "exercise":
            return TextBuilder.build_exercise(payload)

        if domain == "lifestyle":
            return TextBuilder.build_lifestyle(payload)

        return ""

    # ================= UPSERT =================

    def upsert(self, domain, batch):

        if domain == "food":
            self.qdrant.upsert_food_text_batch(batch)

        elif domain == "beverage":
            self.qdrant.upsert_beverage_batch(batch)

        elif domain == "exercise":
            self.qdrant.upsert_exercise_batch(batch)

        elif domain == "lifestyle":
            self.qdrant.upsert_lifestyle_batch(batch)