import numpy as np
from core.services.clip_service import CLIPService
from core.services.qdrant_service import QdrantService


class RAGService:

    def __init__(self):
        self.clip = CLIPService()
        self.qdrant = QdrantService()

    def retrieve(self, query, top_k=20):

        vector = self.clip.embed_text(query)

        results = self.qdrant.client.search(
            collection_name="food_text_vectors",
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )

        return results

    def simple_filter(self, results, max_calories=None):

        filtered = []

        for r in results:
            payload = r.payload

            if max_calories:
                cal = payload.get("calories") or payload.get("Calories")
                if cal and cal > max_calories:
                    continue

            filtered.append(r)

        return filtered

    def build_context(self, results):

        context = ""

        for r in results[:5]:
            context += str(r.payload) + "\n\n"

        return context

    def generate_prompt(self, query, context):

        return f"""
You are a nutrition expert.

User query:
{query}

Relevant food data:
{context}

Answer with:
- specific food recommendations
- calories, protein, carbs
- explanation

Do NOT hallucinate.
"""

    def run(self, query):

        results = self.retrieve(query)

        results = self.simple_filter(results, max_calories=500)

        context = self.build_context(results)

        prompt = self.generate_prompt(query, context)

        return prompt