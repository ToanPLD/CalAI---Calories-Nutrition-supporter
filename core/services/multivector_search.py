from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService

clip = CLIPService()
qdrant = QdrantService()


class MultiVectorSearch:

    def search(self, query, image_path=None, top_k=10):

        text_vec = clip.embed_text(query).tolist()

        text_results = qdrant.client.search(
            collection_name="hf_food_text",
            query_vector=text_vec,
            limit=top_k,
            with_payload=True
        )

        image_results = []

        if image_path:
            img_vec = clip.embed_image(image_path)

            image_results = qdrant.client.search(
                collection_name="hf_food_image",
                query_vector=img_vec,
                limit=top_k,
                with_payload=True
            )

        # 🔥 fusion
        combined = text_results + image_results

        # deduplicate
        seen = set()
        final = []

        for r in combined:
            key = str(r.payload)
            if key not in seen:
                seen.add(key)
                final.append(r)

        return final[:top_k]