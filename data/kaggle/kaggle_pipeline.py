# import os
# import pandas as pd
# import kagglehub

# from core.services.clip_service import CLIPService
# from core.services.qdrant_service import QdrantService
# from core.utils.unit_normalizer import UnitNormalizer
# from core.utils.text_builder import TextBuilder
# # from data.kaggle.dataset_registry import DATASETS


# class KaggleIngestionPipeline:

#     def __init__(self):
#         self.clip = CLIPService()
#         self.qdrant = QdrantService()

#     def load_dataset(self, name):

#         path = kagglehub.dataset_download(name)

#         for root, _, files in os.walk(path):
#             for f in files:
#                 if f.endswith(".csv"):
#                     file_path = os.path.join(root, f)
#                     print("📂 Using:", file_path)
#                     return pd.read_csv(file_path, on_bad_lines="skip")

#         raise ValueError("❌ No CSV found")

#     def run(self):

#         for ds in DATASETS:
#             print(f"\n🚀 {ds['name']}")

#             df = self.load_dataset(ds["name"])

#             batch = []

#             for _, row in df.iterrows():

#                 payload = row.to_dict()

#                 text = TextBuilder.build_generic(payload)

#                 vector = self.clip.embed_text(text)

#                 if vector is None:
#                     continue

#                 batch.append({
#                     "vector": vector.tolist(),
#                     "payload": payload
#                 })

#                 if len(batch) >= 64:
#                     self.qdrant.upsert(f"{ds['domain']}_vectors", batch)
#                     batch.clear()

#             if batch:
#                 self.qdrant.upsert(f"{ds['domain']}_vectors", batch)

#             print("✅ DONE")