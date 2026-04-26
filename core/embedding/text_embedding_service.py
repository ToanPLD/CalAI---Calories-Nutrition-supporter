from sentence_transformers import SentenceTransformer
import torch


class TextEmbeddingService:

    def __init__(self):
        print("🔥 Loading 768 embedding model (mpnet)...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device
        )

    def _clean_text(self, text: str):
        if not text:
            return ""

        return text[:300]

    def embed(self, text: str):

        if not text:
            return None

        try:
            vector = self.model.encode(
                text,
                normalize_embeddings=True,
            )

            return vector.tolist()

        except Exception as e:
            print("❌ Embed error:", e)
            return None

    def embed_batch(self, texts: list[str]):

        if not texts:
            return []

        try:
            vectors = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            return [v.tolist() for v in vectors]

        except Exception as e:
            print("❌ Batch embed error:", e)
            return []