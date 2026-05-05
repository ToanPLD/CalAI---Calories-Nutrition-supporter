import hashlib
import json
import re
import unicodedata
from collections import OrderedDict

from config.settings import settings
from core.embedding.text_embedding_service import TextEmbeddingService
from core.services.cache.redis_cache import RedisCache
from core.services.llm.llm_service import LLMService
from core.services.rag.recipe_image_rag_service import RecipeImageRAGService
from core.services.retrieval.qdrant_service import QdrantService


class AgenticTrace:
    def __init__(self):
        self.steps = []

    def add(self, title, text, status="done", evidence=None, detail=None):
        self.steps.append({
            "step": len(self.steps) + 1,
            "title": title,
            "text": text,
            "status": status,
            "evidence": evidence or [],
            "detail": detail
        })


TOPIC_REGISTRY = {
    "beverage": {
        "phrases": (
            "do uong", "thuc uong", "nuoc uong", "nuoc giai khat",
            "thuc uong giai khat", "loai nuoc", "cac loai nuoc",
            "drink", "drinks", "beverage", "beverages",
            "smoothie", "smoothies", "juice", "juices",
            "tra", "cafe", "coffee", "tea", "soda", "milk",
            "sua tuoi", "sua dau", "sua hat", "nuoc ep",
        ),
        "keywords": ("drink", "beverage", "tea", "coffee", "juice", "milk", "smoothie"),
        "collections": ("beverage_text_vectors_768", "beverage_vectors_768"),
        "exclusive": True,
    },
    "exercise": {
        "phrases": (
            "bai tap", "bai the duc", "tap luyen", "the duc", "the thao",
            "van dong", "hoat dong the chat", "tap gym", "tap aerobic",
            "tap yoga", "chay bo", "boi loi", "dap xe", "nang ta",
            "exercise", "exercises", "workout", "workouts", "training",
            "fitness", "gym", "sport", "sports", "physical activity",
            "cardio", "yoga", "running", "swimming", "cycling",
            "weightlifting", "strength training",
        ),
        "keywords": (
            "exercise", "workout", "training", "fitness", "gym",
            "sport", "activity", "cardio", "running", "yoga",
        ),
        "collections": (
            "exercise_text_vectors_768",
            "exercise_vectors_768",
            "exercise_gym_vectors_768",
        ),
        "exclusive": True,
    },
    "obesity_lifestyle": {
        "phrases": (
            "loi song", "thoi quen sinh hoat", "thoi quen an uong",
            "beo phi", "thua can", "obesity", "overweight",
            "lifestyle", "habit", "habits", "smoking",
            "alcohol", "screen time", "sedentary",
        ),
        "keywords": (
            "lifestyle", "habit", "obesity", "overweight",
            "smoking", "alcohol", "sedentary",
        ),
        "collections": ("lifestyle_obesity_vectors_768", "lifestyle_vectors_768"),
        "exclusive": True,
    },
    "diet_disease": {
        "phrases": (
            "che do an cho", "che do dinh duong cho", "an cho nguoi",
            "tieu duong", "huyet ap cao", "huyet ap thap", "tim mach",
            "ung thu", "benh nhan", "benh ly",
            "diabetes", "diabetic", "hypertension", "heart disease",
            "kidney disease", "cholesterol", "dietary recommendation",
            "dietary advice", "patient diet",
        ),
        "keywords": (
            "diet", "diabetes", "hypertension", "heart disease",
            "kidney", "cholesterol", "patient", "recommendation",
        ),
        "collections": ("diet_recommendations_vectors",),
        "exclusive": False,
    },
    "fruit": {
        "phrases": (
            "trai cay", "hoa qua", "fruit", "fruits",
        ),
        "keywords": ("fruit", "apple", "banana", "orange", "guava", "mango"),
        "collections": ("food_fruit_vectors_768",),
        "exclusive": False,
    },
}

BEVERAGE_PHRASES = TOPIC_REGISTRY["beverage"]["phrases"]


def _normalize_for_topic(text):
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    return text


def _topic_phrase_matches(normalized, phrases):
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", normalized)
        for phrase in phrases
    )


def matched_topics(query):
    normalized = _normalize_for_topic(query)
    return [
        topic for topic, spec in TOPIC_REGISTRY.items()
        if _topic_phrase_matches(normalized, spec["phrases"])
    ]


class AgentRouter:
    def _normalize(self, text):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("đ", "d").replace("Đ", "D")
        return text.lower()

    def _has_phrase(self, text, phrases):
        for phrase in phrases:
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        return False

    def classify(self, query, forced_intent=None):
        if forced_intent:
            return forced_intent

        q = self._normalize(query)

        if self._has_phrase(q, [
            "tang bao nhieu kg", "giam bao nhieu kg", "tang can bao nhieu",
            "giam can bao nhieu", "se tang bao nhieu", "se giam bao nhieu",
            "can nang hien tai", "tdee", "bmr", "surplus", "deficit",
            "thang du calo", "thieu hut calo", "calorie surplus",
            "calorie deficit", "energy balance", "kg fat", "kg mo"
        ]):
            return "weight_projection"

        if self._has_phrase(q, [
            "so sanh", "compare", "comparison", "vs", "versus",
            "khac nhau", "nguyen lieu nao", "bang so sanh"
        ]):
            return "ingredient_comparison"

        if self._has_phrase(q, [
            "bien tau", "goi y bien", "thay the", "substitute",
            "multi hop", "ket hop", "extract ingredients", "lay nguyen lieu"
        ]):
            return "multi_hop"

        if self._has_phrase(q, [
            "tim anh", "hinh anh", "anh mon", "image", "photo", "picture"
        ]):
            return "image_retrieval"

        topics = matched_topics(query)
        if topics and any(TOPIC_REGISTRY[topic].get("exclusive") for topic in topics):
            return "nutrition_qa"

        if self._has_phrase(q, [
            "plan my lunch", "plan lunch", "plan my dinner", "plan dinner",
            "plan breakfast", "meal plan", "lunch plan", "dinner plan",
            "breakfast plan", "lap thuc don", "len thuc don", "thuc don",
            "lap ke hoach", "len ke hoach", "ke hoach bua an",
            "ke hoach an uong", "bua an hom nay", "hom nay an gi",
            "an gi hom nay", "daily meal plan", "plan my meals",
            "bua trua", "bua toi", "bua sang", "an trua", "an toi",
            "an sang", "menu", "meal prep"
        ]):
            return "meal_planning"

        if (
            self._has_phrase(q, ["liet ke", "danh sach", "goi y", "de xuat", "suggest", "recommend"])
            and self._has_phrase(q, ["mon", "bua", "bua trua", "bua toi", "bua sang", "lunch", "dinner", "breakfast", "giau protein", "high protein"])
        ):
            return "meal_planning"

        if topics:
            return "nutrition_qa"

        if self._has_phrase(q, [
            "calo", "kcal", "calorie", "macro", "protein", "carb", "fat",
            "dinh duong", "nutrition", "giam can", "tang can", "diet",
            "fitness", "tap luyen", "how many calories", "calories in",
            "bao nhieu calo", "bao nhieu kcal", "toi an", "minh an",
            "vua an", "dang an", "i ate", "i am eating"
        ]):
            return "nutrition_qa"

        if self._has_phrase(q, [
            "cong thuc", "recipe", "cach lam", "instructions",
            "nguyen lieu", "mon nay nau", "nau mon", "liet ke",
            "danh sach", "bang", "table", "bieu do", "chart", "so lieu"
        ]):
            return "recipe_reasoning"

        return "general_rag"


class CitationBuilder:
    TITLE_FIELDS = (
        "title", "Title", "recipe_title", "Recipe_Title",
        "recipe_name", "Recipe_Name", "Recipe Name", "recipeName",
        "name", "Name",
        "dish_name", "Dish_Name", "dish",
        "food_name", "Food_Name", "Food", "Food_Item", "food",
        "product_name", "Product_Name", "product",
        "drink", "Drink", "Drink_Name", "beverage_name", "Beverage",
        "exercise_name", "Exercise", "Activity", "activity_name",
        "Disease", "disease_name", "Habit", "habit_name",
        "Fruit", "Vegetable",
        "Shrt_Desc", "shrt_desc", "Long_Desc",
    )

    @staticmethod
    def from_payload(payload):
        payload = payload or {}
        explicit = payload.get("citation")
        if isinstance(explicit, dict):
            return explicit

        dataset = payload.get("source_dataset") or payload.get("domain")
        title = None
        for field in CitationBuilder.TITLE_FIELDS:
            value = payload.get(field)
            if value not in (None, ""):
                title = value
                break

        return {
            "dataset": dataset,
            "collection": payload.get("source_collection"),
            "row": payload.get("source_row") or payload.get("ref_id"),
            "title": title,
            "image_name": payload.get("image_name"),
            "image_file": payload.get("image_file")
        }

    @staticmethod
    def display_label(payload, fallback=None):
        citation = CitationBuilder.from_payload(payload)
        title = citation.get("title")
        if title not in (None, ""):
            text = str(title).strip()
            if text:
                return text[:80]
        collection = citation.get("collection") or citation.get("dataset")
        row = citation.get("row")
        if collection and row not in (None, ""):
            return f"{collection}#{row}"
        if collection:
            return str(collection)
        return "" if fallback is None else str(fallback)

    @staticmethod
    def dedupe(citations):
        unique = OrderedDict()
        for citation in citations:
            if not citation:
                continue
            key = (
                citation.get("dataset"),
                citation.get("collection"),
                citation.get("row"),
                citation.get("title"),
                citation.get("image_name"),
            )
            unique[key] = citation
        return list(unique.values())


class AgenticResponseGenerator:
    def __init__(self, llm=None):
        self.llm = llm or LLMService()

    def _model_unavailable_answer(self):
        return (
            "Xin lỗi bạn, mình đang gặp trục trặc khi xử lý câu hỏi này. "
            "Bạn thử gửi lại nhé, hoặc bổ sung thêm chi tiết "
            "(ví dụ: tên món cụ thể, khẩu phần, mục tiêu dinh dưỡng) "
            "để mình hỗ trợ tốt hơn!"
        )

    def _no_context_answer(self, query):
        return (
            f"Mình chưa tìm thấy dữ liệu phù hợp trong kho thông tin cho câu hỏi của bạn. "
            "Bạn có thể thử:\n"
            "- Mô tả cụ thể hơn tên món ăn hoặc thực phẩm bạn muốn tra cứu\n"
            "- Dùng tên tiếng Anh nếu là món quốc tế (ví dụ: chicken breast, salmon)\n"
            "- Hỏi về một loại thực phẩm cụ thể thay vì câu hỏi chung\n\n"
            "Mình sẵn sàng giúp bạn ngay khi có thêm thông tin!"
        )

    async def generate(self, query, intent, context, citations, trace, conversation_context=None, user_profile_text=None):
        if not context and intent not in ("weight_projection",):
            return self._no_context_answer(query)

        try:
            answer = await self.llm.answer_agentic(
                query=query,
                intent=intent,
                context=context,
                citations=citations,
                conversation_context=conversation_context,
                user_profile_text=user_profile_text
            )
        except Exception:
            return self._model_unavailable_answer()

        return answer or self._model_unavailable_answer()


class GenericRAGAgent:
    FOOD_SYNONYMS = {
        "tao": "apple",
        "qua tao": "apple",
        "chuoi": "banana",
        "cam": "orange",
        "com": "rice",
        "gao": "rice",
        "uc ga": "chicken breast",
        "thit ga": "chicken",
        "thit bo": "beef",
        "ca hoi": "salmon",
        "trung": "egg",
        "sua chua": "yogurt",
        "khoai lang": "sweet potato",
        "banh mi": "bread sandwich",
        "pizza": "pizza",
        "salad": "salad",
        "pasta": "pasta",
        "pho": "pho beef noodle soup",
        "bun bo": "beef noodle soup",
        "mi": "noodle",
        "yen mach": "oats",
        "dau phu": "tofu",
    }

    def __init__(self, qdrant=None, text_embed=None):
        self.qdrant = qdrant or QdrantService()
        self.text_embed = text_embed or TextEmbeddingService()
        self._compatible_collections = None

    def _text_collections(self):
        if self._compatible_collections is not None:
            return self._compatible_collections

        available = self.qdrant.available_collections()

        configured = [
            collection for collection in settings.TEXT_COLLECTIONS
            if collection in available
        ]
        discovered = []
        for collection in available:
            if collection in configured:
                continue
            try:
                info = self.qdrant.client.get_collection(collection)
                vectors = info.config.params.vectors
                if isinstance(vectors, dict):
                    continue
                if getattr(vectors, "size", None) == settings.TEXT_VECTOR_DIM:
                    discovered.append(collection)
            except Exception:
                continue

        merged = []
        for collection in [*configured, *sorted(discovered)]:
            if collection not in merged:
                merged.append(collection)

        if not merged:
            print("⚠️ No compatible Qdrant collections found. Available:", available or "none")

        self._compatible_collections = merged or settings.TEXT_COLLECTIONS
        return self._compatible_collections

    def _nutrition_collections(self):
        existing = set(self._text_collections())
        preferred = [
            "food_fruit_vectors_768",
            "food_common_vectors_768",
            "food_nutrition_vectors_768",
            "food_nutrition_dev_vectors_768",
            "food_global_10k_vectors_768",
            "food_text_vectors_768",
            "food_vectors_768",
            "beverage_text_vectors_768",
            "beverage_vectors_768",
        ]
        selected = [collection for collection in preferred if collection in existing]
        return selected or self._text_collections()

    def _focused_nutrition_collections(self, query):
        existing = set(self._text_collections())
        keywords = set(self._query_keywords(query))
        topics = matched_topics(query)

        topic_collections = []
        any_exclusive = False
        for topic in topics:
            spec = TOPIC_REGISTRY[topic]
            topic_collections.extend(spec["collections"])
            if spec.get("exclusive"):
                any_exclusive = True

        if topic_collections:
            preferred = list(topic_collections)
            if not any_exclusive:
                preferred += [
                    "food_nutrition_vectors_768",
                    "food_common_vectors_768",
                ]
            selected = [c for c in preferred if c in existing]
            return list(dict.fromkeys(selected)) or self._nutrition_collections()

        fruit_keywords = {"apple", "banana", "orange"}
        beverage_keywords = {"milk", "coffee", "tea", "juice", "smoothie"}
        if keywords & beverage_keywords:
            preferred = [
                "beverage_text_vectors_768",
                "beverage_vectors_768",
                "food_nutrition_vectors_768",
                "food_common_vectors_768",
            ]
        elif keywords & fruit_keywords:
            preferred = [
                "food_fruit_vectors_768",
                "food_common_vectors_768",
                "food_nutrition_vectors_768",
            ]
        elif keywords:
            preferred = [
                "food_common_vectors_768",
                "food_nutrition_vectors_768",
                "food_nutrition_dev_vectors_768",
                "food_global_10k_vectors_768",
                "food_vectors_768",
                "food_text_vectors_768",
            ]
        else:
            return self._nutrition_collections()

        selected = [collection for collection in preferred if collection in existing]
        return selected or self._nutrition_collections()

    def _expand_query(self, query):
        keywords = self._query_keywords(query)
        if not keywords:
            return query

        return f"{query}\nEnglish retrieval keywords: {', '.join(dict.fromkeys(keywords))}"

    def _query_keywords(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").lower()

        keywords = []
        for source, target in self.FOOD_SYNONYMS.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(source) + r"(?![a-z0-9])", normalized):
                keywords.append(target)
        for target in sorted(set(self.FOOD_SYNONYMS.values()), key=len, reverse=True):
            if re.search(r"(?<![a-z0-9])" + re.escape(target.lower()) + r"(?![a-z0-9])", normalized):
                keywords.append(target)

        for topic in matched_topics(query):
            keywords.extend(TOPIC_REGISTRY[topic]["keywords"])

        return list(dict.fromkeys(keywords))

    def _comparison_terms(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").replace("Đ", "D").lower()

        match = re.search(
            r"(?:so sanh|compare|comparison|khac nhau giua|khac nhau)\s+(.+)",
            normalized
        )
        segment = match.group(1) if match else normalized
        segment = re.sub(
            r"\b(?:duoi dang|dang)?\s*(?:bang|table|bieu do|chart)\b.*",
            " ",
            segment
        )
        segment = re.sub(
            r"\b(?:cho nguoi|cho|minh|toi|de|phu hop voi|dang)?\s*(?:giam can|tang can|diet|eat clean)\b.*",
            " ",
            segment
        )
        segment = re.sub(
            r"\b(?:dinh duong|nutrition|nutrient|macro|macros|calo|calorie|calories|"
            r"kcal|protein|carb|fat|fiber|chat beo|chat xo|nguyen lieu|thuc pham|"
            r"mon an|mon|food|ingredient|cac|loai|giua|nao|hon|tot hon|nen an|"
            r"doi tuong|bat ki|bat ky|bat cu|any|random|giau|bo duong)\b",
            " ",
            segment
        )
        segment = re.sub(r"\b\d+\b", " ", segment)
        segment = re.sub(r"\s+", " ", segment).strip()

        parts = re.split(r"\s+(?:va|voi|and|vs|versus)\s+|[,/]+", segment)
        terms = []
        for part in parts:
            term = re.sub(r"\s+", " ", part).strip(" .:;!?-")
            if len(term) < 2:
                continue
            if term not in terms:
                terms.append(term[:80])
        return terms[:4]

    def _display_name(self, payload):
        return str(
            payload.get("title")
            or payload.get("Name")
            or payload.get("recipe_name")
            or payload.get("name")
            or payload.get("dish_name")
            or payload.get("food_name")
            or payload.get("product_name")
            or payload.get("food")
            or payload.get("Fruit")
            or payload.get("Shrt_Desc")
            or ""
        )

    def _rerank_score(self, hit, keywords):
        score = float(getattr(hit, "score", 0) or 0)
        payload = hit.payload or {}
        if not keywords:
            return score

        name = unicodedata.normalize("NFKD", self._display_name(payload))
        name = "".join(ch for ch in name if not unicodedata.combining(ch))
        name = name.replace("đ", "d").lower()
        collection = str(payload.get("source_collection") or "")

        bonus = 0
        for keyword in keywords:
            keyword = keyword.lower()
            if name == keyword:
                bonus += 0.35
            elif re.search(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])", name):
                bonus += 0.12
        if collection == "food_fruit_vectors_768":
            bonus += 0.08

        return score + bonus

    def _search_hits(self, query, top_k, collections=None, per_collection=None):
        expanded_query = self._expand_query(query)
        keywords = self._query_keywords(query)

        vector = self.text_embed.embed(expanded_query)
        if vector is None:
            return []

        collections = collections or self._text_collections()
        hits = []
        per_collection = per_collection or max(1, min(4, top_k))
        for collection in collections:
            for hit in self.qdrant.search(
                collection_name=collection,
                vector=vector,
                top_k=per_collection
            ):
                payload = dict(hit.payload or {})
                payload.setdefault("source_collection", collection)
                hit.payload = payload
                hits.append(hit)

        hits.sort(key=lambda hit: self._rerank_score(hit, keywords), reverse=True)
        return hits[:top_k]

    def run(self, query, top_k, trace, collections=None):
        trace.add(
            "Generic RAG retrieval",
            "Embed query và tìm trong các collection text/nutrition/lifestyle hiện có.",
            detail="Bỏ qua collection không tồn tại trong Qdrant để giảm lỗi 404."
        )
        expanded_query = self._expand_query(query)
        keywords = self._query_keywords(query)
        if expanded_query != query:
            trace.add(
                "Query expansion",
                "Mở rộng query tiếng Việt sang keyword tiếng Anh để khớp dataset Qdrant.",
                evidence=keywords[:8]
            )

        selected = self._search_hits(query, top_k, collections=collections)
        if not selected:
            trace.add("Embedding failed", "Không tạo được vector query hoặc không có kết quả.", status="warning")
            return []

        collections = collections or self._text_collections()
        trace.add(
            "Generic RAG selected context",
            f"Chọn {len(selected)} context tốt nhất từ {len(collections)} collection khả dụng.",
            evidence=[CitationBuilder.display_label(hit.payload, fallback=hit.id) for hit in selected[:5]]
        )
        return [
            {
                "score": hit.score,
                "payload": hit.payload or {},
                "citation": CitationBuilder.from_payload(hit.payload or {})
            }
            for hit in selected
        ]

    def ingredient_comparison(self, query, top_k, trace):
        terms = self._comparison_terms(query)
        trace.add(
            "Ingredient comparison retrieval",
            "Tách từng thực phẩm/nguyên liệu rồi truy vấn các collection dinh dưỡng và text 768 chiều.",
            evidence=terms
        )

        if not terms:
            trace.add(
                "Open-ended comparison detected",
                "Không có thực phẩm cụ thể để truy vấn Qdrant; response generator sẽ hỏi rõ đối tượng hoặc nêu giả định thay vì semantic search nhiễu.",
                status="skipped"
            )
            return []

        buckets = []
        per_term = max(3, min(5, top_k))
        for term in terms:
            expanded_term = self._expand_query(term)
            trace.add(
                "Comparison sub-query",
                f"Tìm dữ liệu dinh dưỡng gần nhất cho `{term}`.",
                evidence=[expanded_term]
            )
            collections = self._focused_nutrition_collections(term)
            bucket = []
            for hit in self._search_hits(
                term,
                per_term,
                collections=collections,
                per_collection=2
            ):
                payload = dict(hit.payload or {})
                payload["comparison_term"] = term
                bucket.append({
                    "score": hit.score,
                    "payload": payload,
                    "comparison_term": term,
                    "citation": CitationBuilder.from_payload(payload)
                })
            buckets.append(bucket)

        merged = OrderedDict()
        cursor = 0
        while len(merged) < top_k and any(cursor < len(bucket) for bucket in buckets):
            for bucket in buckets:
                if cursor >= len(bucket):
                    continue
                item = bucket[cursor]
                payload = item.get("payload") or {}
                key = (
                    item.get("comparison_term"),
                    payload.get("source_collection"),
                    payload.get("source_row"),
                    payload.get("Name") or payload.get("name") or payload.get("title")
                )
                if key not in merged:
                    merged[key] = item
                if len(merged) >= top_k:
                    break
            cursor += 1

        results = list(merged.values())
        trace.add(
            "Comparison context prepared",
            f"Chuẩn hóa {len(results)} dòng dữ liệu để response generator tạo bảng so sánh.",
            evidence=[
                str((item.get("payload") or {}).get("Name") or (item.get("payload") or {}).get("name") or item.get("comparison_term"))
                for item in results[:5]
            ]
        )
        return results


class RecipeAgent:
    def __init__(self, recipe_rag=None):
        self.rag = recipe_rag or RecipeImageRAGService()

    def _food_keywords(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").replace("Đ", "D").lower()
        keywords = []
        for source, target in GenericRAGAgent.FOOD_SYNONYMS.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(source) + r"(?![a-z0-9])", normalized):
                keywords.append(target)
        for target in sorted(set(GenericRAGAgent.FOOD_SYNONYMS.values()), key=len, reverse=True):
            if re.search(r"(?<![a-z0-9])" + re.escape(target.lower()) + r"(?![a-z0-9])", normalized):
                keywords.append(target)
        return list(dict.fromkeys(keywords))

    def _ingredient_hint(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").replace("Đ", "D")
        normalized = normalized.lower()
        keywords = self._food_keywords(query)
        if len(keywords) == 1:
            return keywords[0]
        if len(keywords) > 1:
            return None

        match = re.search(r"(?:co|voi|with|ingredient)\s+([a-z0-9\-\s]+)", normalized)
        if not match:
            return None
        hint = re.split(
            r",|\\. |\\?|\\b(?:goi y|bien tau|cong thuc|recipe|de|for|please)\\b",
            match.group(1)
        )[0]
        return hint.strip()[:80] or None

    def _recipe_search_query(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").replace("Đ", "D")
        normalized = normalized.lower()
        normalized = re.sub(
            r"\\b(?:cho toi|hay|please|toi muon|can|tim|cong thuc|recipe|"
            r"cach lam|instructions|nau|mon|món|giup toi|cho minh)\\b",
            " ",
            normalized
        )
        normalized = re.sub(r"\\s+", " ", normalized).strip(" .:;!?-")
        if not normalized:
            return query
        keywords = self._food_keywords(query)
        keyword_text = f"\nEnglish ingredient keywords: {', '.join(keywords)}" if keywords else ""
        return f"recipe {normalized}{keyword_text}"

    def _recipe_match_score(self, hit, keywords):
        score = float(getattr(hit, "score", 0) or 0)
        if not keywords:
            return score

        payload = hit.payload or {}
        text = " ".join([
            str(payload.get("title") or ""),
            str(payload.get("recipe_name") or ""),
            str(payload.get("ingredients_search") or ""),
            str(payload.get("cleaned_ingredients") or ""),
            str(payload.get("ingredients") or ""),
            str(payload.get("directions") or ""),
            str(payload.get("image_caption") or ""),
        ]).lower()

        bonus = 0
        for keyword in keywords:
            keyword = keyword.lower()
            if re.search(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])", text):
                bonus += 0.18
            elif keyword in text:
                bonus += 0.08
        return score + bonus

    def _comparison_terms(self, query):
        normalized = unicodedata.normalize("NFKD", str(query or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace("đ", "d").replace("Đ", "D")
        normalized = normalized.lower()

        match = re.search(
            r"(?:so sanh|compare|comparison|khac nhau giua)\s+(.+)",
            normalized
        )
        segment = match.group(1) if match else normalized
        segment = re.sub(
            r"\b(mon|cac mon|món|dish|food|nguyen lieu|thuc pham|giua|loai)\b",
            " ",
            segment
        )
        parts = re.split(r"\s+(?:va|voi|and|vs|versus)\s+|[,/]+", segment)

        terms = []
        for part in parts:
            term = re.sub(r"\s+", " ", part).strip(" .:;!?-")
            if len(term) < 2:
                continue
            if term not in terms:
                terms.append(term[:80])
        return terms[:4]

    RECIPE_TEXT_COLLECTIONS = ("recipes_vectors_768", "food_recipes_vectors_768")

    def _parse_ingredient_list(self, value):
        if isinstance(value, list):
            return value
        if not value:
            return []
        text = str(value)
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [part.strip() for part in re.split(r",\s*", text) if part.strip()]

    def _search_recipe_text_collections(self, query, top_k):
        vector = self.rag.text_embed.embed(query)
        if vector is None:
            return []

        hits = []
        for collection in self.RECIPE_TEXT_COLLECTIONS:
            try:
                results = self.rag.client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True
                )
            except Exception as exc:
                print(f"❌ Recipe text search error ({collection}):", exc)
                continue
            for hit in results:
                payload = dict(hit.payload or {})
                payload.setdefault("source_collection", collection)
                hit.payload = payload
                hits.append(hit)
        return hits

    def _format_text_recipe_hits(self, hits):
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "score": hit.score,
                "title": payload.get("recipe_name") or payload.get("title"),
                "ingredients": self._parse_ingredient_list(payload.get("ingredients")),
                "instructions": payload.get("directions") or payload.get("instructions"),
                "image_name": None,
                "image_file": None,
                "image_path": None,
                "image_caption": None,
                "citation": CitationBuilder.from_payload(payload),
                "payload": payload
            })
        return results

    def recipe_reasoning(self, query, top_k, trace):
        ingredient = self._ingredient_hint(query)
        search_query = self._recipe_search_query(query)
        keywords = self._food_keywords(query)
        trace.add(
            "Recipe reasoning retrieval",
            "Tìm công thức bằng vector text 768 chiều trên multimodal + recipe text collections.",
            evidence=[f"search_query={search_query}"] + ([f"ingredient_filter={ingredient}"] if ingredient else [])
        )

        over_fetch = max(top_k, top_k * 4 if keywords else top_k)
        primary_hits = list(self.rag.search_text(
            query=search_query,
            top_k=over_fetch,
            ingredient=ingredient
        ))
        text_hits = self._search_recipe_text_collections(search_query, over_fetch)

        combined = primary_hits + text_hits
        combined.sort(
            key=lambda hit: self._recipe_match_score(hit, keywords),
            reverse=True
        )
        combined = combined[:top_k]

        results = []
        for hit in combined:
            collection = (hit.payload or {}).get("source_collection")
            if collection in self.RECIPE_TEXT_COLLECTIONS:
                results.extend(self._format_text_recipe_hits([hit]))
            else:
                results.extend(self.rag.format_hits([hit]))
        trace.add(
            "Recipe context selected",
            f"Chọn {len(results)} công thức liên quan để đưa vào response generator.",
            evidence=[item.get("title") for item in results[:5] if item.get("title")],
            detail="Rerank ưu tiên công thức chứa nguyên liệu chính đã nhận diện trong câu hỏi."
        )
        return results

    def image_retrieval(self, query, top_k, trace):
        ingredient = self._ingredient_hint(query)
        trace.add(
            "Image retrieval",
            "Tìm ảnh công thức bằng named vector `image` 512 chiều và metadata filter nếu có.",
            evidence=[f"ingredient_filter={ingredient}"] if ingredient else []
        )
        hits = self.rag.search_images(
            query=query,
            top_k=top_k,
            ingredient=ingredient
        )
        results = self.rag.format_hits(hits)
        trace.add(
            "Image candidates selected",
            f"Chọn {len(results)} ảnh/công thức phù hợp nhất.",
            evidence=[item.get("image_file") or item.get("image_name") for item in results[:5]]
        )
        return results

    def ingredient_comparison(self, query, top_k, trace):
        terms = self._comparison_terms(query)
        trace.add(
            "Ingredient comparison retrieval",
            "Tách từng đối tượng cần so sánh rồi tìm context riêng bằng text vector.",
            evidence=terms
        )

        if not terms:
            results = self.recipe_reasoning(query, top_k, trace)
        else:
            buckets = []
            per_term = max(2, min(4, top_k))
            for term in terms:
                trace.add(
                    "Comparison sub-query",
                    f"Tìm dữ liệu liên quan đến `{term}`.",
                    evidence=[term]
                )
                hits = self.rag.search_text(query=term, top_k=per_term)
                bucket = []
                for item in self.rag.format_hits(hits):
                    item["comparison_term"] = term
                    bucket.append(item)
                buckets.append(bucket)

            merged = OrderedDict()
            cursor = 0
            while len(merged) < top_k and any(cursor < len(bucket) for bucket in buckets):
                for bucket in buckets:
                    if cursor >= len(bucket):
                        continue
                    item = bucket[cursor]
                    payload = item.get("payload") or {}
                    key = payload.get("source_row") or item.get("title") or f"{item.get('comparison_term')}:{len(merged)}"
                    if key not in merged:
                        merged[key] = item
                    if len(merged) >= top_k:
                        break
                cursor += 1
            results = list(merged.values())

        trace.add(
            "Comparison context prepared",
            "Chuẩn hóa danh sách nguyên liệu để response generator tạo bảng so sánh.",
            evidence=[item.get("title") for item in results[:5] if item.get("title")]
        )
        return results

    def multi_hop(self, query, top_k, trace):
        trace.add(
            "Multi-hop step 1",
            "Tìm ảnh/recipe candidates trước để lấy món và citation gốc."
        )
        image_results = self.image_retrieval(query, max(2, top_k // 2), trace)

        ingredient_terms = []
        for item in image_results:
            ingredients = item.get("ingredients") or []
            if isinstance(ingredients, list):
                ingredient_terms.extend(ingredients[:4])

        enriched_query = " ".join([
            query,
            " ".join(ingredient_terms[:12])
        ]).strip()
        trace.add(
            "Multi-hop step 2",
            "Extract nguyên liệu từ kết quả bước 1 và mở rộng query để tìm công thức/biến tấu.",
            evidence=ingredient_terms[:8]
        )

        recipe_hits = self.rag.search_text(
            query=enriched_query,
            top_k=top_k,
        )
        recipe_results = self.rag.format_hits(recipe_hits)

        merged = OrderedDict()
        for item in image_results + recipe_results:
            key = item.get("payload", {}).get("source_row") or item.get("title")
            merged[key] = item

        results = list(merged.values())[:top_k]
        trace.add(
            "Multi-hop context merged",
            f"Gộp {len(results)} kết quả từ image retrieval và recipe reasoning.",
            evidence=[item.get("title") for item in results[:5] if item.get("title")]
        )
        return results


class AgenticRAG:
    def __init__(self):
        self.router = AgentRouter()
        self.recipe_agent = RecipeAgent()
        self.generic_agent = None
        self.response_generator = AgenticResponseGenerator()
        self.cache = RedisCache()

    def _generic_agent(self):
        if self.generic_agent is None:
            self.generic_agent = GenericRAGAgent()
        return self.generic_agent

    def _context_from_results(self, results):
        context = []
        for result in results:
            payload = dict(result.get("payload") or result)
            if result.get("comparison_term"):
                payload["comparison_term"] = result["comparison_term"]
            if payload:
                context.append(payload)
        return context

    def _is_pure_affirmation(self, query):
        normalized = self.router._normalize(query).strip()
        if not normalized or len(normalized) > 12:
            return False
        affirmations = {
            "co", "yes", "ya", "yeah", "yep", "ok", "okay", "okie",
            "u", "um", "uhm", "vang", "da", "duoc", "duoc nha",
            "duoc roi", "co nhe", "yes please", "lam di", "go", "go ahead",
        }
        return normalized in affirmations

    def _is_vague_followup(self, query, is_follow_up, conversation_context):
        if not conversation_context or not is_follow_up:
            return False
        normalized = self.router._normalize(query)
        if len(normalized) > 60:
            return False

        suggest_phrases = [
            "goi y", "de xuat", "co the goi y", "co the de xuat",
            "suggest", "recommend", "co the cho", "cho toi them",
            "noi them", "noi ro hon", "them thong tin", "co the noi",
            "gi khac", "khac khong", "co gi khac", "tuong tu",
            "lam thu", "thu xem", "lam giup", "lam cho toi",
            "thu cho toi", "hay lam thu", "hay lam", "lam di",
            "try it", "show me", "do it", "give me one",
        ]
        has_suggest = any(phrase in normalized for phrase in suggest_phrases)
        if not has_suggest:
            return False

        topic_terms = list(GenericRAGAgent.FOOD_SYNONYMS.keys()) + [
            "calo", "kcal", "calorie", "macro", "protein", "carb", "fat",
            "fiber", "chat xo", "tdee", "bmr", "bua trua", "bua toi",
            "bua sang", "lunch", "dinner", "breakfast", "thuc don",
            "meal plan", "diet",
        ]
        has_topic = any(
            re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", normalized)
            for term in topic_terms
        )
        return not has_topic

    def _last_assistant_question(self, conversation_context):
        """Pull the final assistant turn's question from conversation context.
        Used to anchor short affirmations like 'có' on what the assistant just asked."""
        if not conversation_context:
            return None
        text = str(conversation_context)
        last_assistant = None
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Assistant:"):
                last_assistant = stripped[len("Assistant:"):].strip()
        if not last_assistant:
            return None
        # Take the last sentence that ends with `?`
        sentences = re.split(r"(?<=[.?!])\s+", last_assistant)
        for sentence in reversed(sentences):
            if "?" in sentence:
                return sentence.strip()
        return last_assistant[-220:]

    def _extract_recent_topic(self, conversation_context):
        if not conversation_context:
            return None
        text = str(conversation_context)
        last_user = None
        last_assistant = None
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("User:"):
                last_user = stripped[len("User:"):].strip()
            elif stripped.startswith("Assistant:"):
                last_assistant = stripped[len("Assistant:"):].strip()
        return last_user or last_assistant

    def _retrieval_query(self, query, conversation_context=None, is_follow_up=None):
        if not conversation_context:
            return query

        if is_follow_up is None:
            normalized = self.router._normalize(query)
            is_follow_up = len(normalized) <= 28 or self.router._has_phrase(normalized, [
                "mon nay", "mon do", "cai nay", "cai do", "no", "nay",
                "tiep", "tinh tiep", "vay con", "so sanh voi", "them",
                "bot", "doi sang", "nhu tren", "nhu vay", "vay trong",
                "luong dinh duong", "can nang hien tai", "tang bao nhieu",
                "giam bao nhieu", "the thi", "this", "that", "it"
            ])

        if not is_follow_up:
            return query

        compact_context = str(conversation_context)[-1600:]
        return (
            "Ngữ cảnh hội thoại gần đây:\n"
            f"{compact_context}\n\n"
            f"Câu hỏi hiện tại: {query}"
        )

    def _cache_key(self, query, top_k, intent, session_id=None, conversation_context=None):
        context_digest = hashlib.sha1(
            str(conversation_context or "")[-1600:].encode("utf-8")
        ).hexdigest()[:12]
        raw = "|".join([
            str(session_id or "global"),
            self.router._normalize(query),
            str(top_k),
            str(intent or ""),
            context_digest
        ])
        return "agentic_rag:v8:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key):
        cached = self.cache.get(key)
        if not cached:
            return None
        try:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return json.loads(cached)
        except Exception:
            return None

    def _cache_set(self, key, response):
        compact = {
            "type": response.get("type"),
            "intent": response.get("intent"),
            "answer": response.get("answer"),
            "citations": response.get("citations", [])[:6],
            "context_used": response.get("context_used", [])[:2],
            "trace": response.get("trace", [])[:8],
            "session_id": response.get("session_id"),
        }
        self.cache.set(
            key,
            json.dumps(compact, ensure_ascii=False),
            ttl=settings.AGENTIC_CACHE_TTL
        )

    MEAL_TYPE_PATTERNS = {
        "breakfast": ("bua sang", "an sang", "buoi sang", "breakfast", "morning meal"),
        "lunch": ("bua trua", "an trua", "buoi trua", "lunch", "midday meal"),
        "dinner": ("bua toi", "an toi", "buoi toi", "dinner", "supper", "evening meal"),
    }

    MEAL_TYPE_SEEDS = {
        "breakfast": (
            "breakfast morning meal oats oatmeal porridge eggs scrambled boiled omelet "
            "Greek yogurt yogurt cottage cheese fruit berries banana apple "
            "whole grain toast pancakes waffles granola muesli milk smoothie protein"
        ),
        "lunch": (
            "lunch midday meal salad sandwich wrap rice bowl quinoa bowl "
            "chicken breast tuna salmon turkey vegetables soup beans lentils "
            "brown rice whole grain pasta protein"
        ),
        "dinner": (
            "dinner evening meal chicken breast salmon lean beef tofu fish shrimp "
            "vegetables broccoli spinach rice quinoa pasta soup stew steamed grilled protein"
        ),
    }

    GENERIC_MEAL_SEED = (
        "meal plan high protein low carb calories protein carbohydrates fat serving "
        "chicken breast egg tuna salmon shrimp lean beef tofu Greek yogurt cottage cheese vegetables salad"
    )

    BEVERAGE_SLOT_SEED = (
        "beverage drink water tea green tea unsweetened tea coffee black coffee milk skim milk "
        "soy milk almond milk oat milk smoothie protein shake fruit juice orange juice"
    )

    HORIZON_NUMBER_PATTERNS = (
        (r"(\d+)\s*ngay", 1),
        (r"(\d+)\s*day", 1),
        (r"(\d+)\s*tuan", 7),
        (r"(\d+)\s*week", 7),
        (r"(\d+)\s*thang", 30),
        (r"(\d+)\s*month", 30),
    )

    HORIZON_WORD_PATTERNS = (
        (r"(?<![a-z0-9])(?:mot|one|ca|whole|nguyen)\s*thang(?![a-z0-9])", 30),
        (r"(?<![a-z0-9])(?:mot|one|ca|whole|nguyen)\s*month(?![a-z0-9])", 30),
        (r"(?<![a-z0-9])(?:mot|one|ca|whole|nguyen)\s*tuan(?![a-z0-9])", 7),
        (r"(?<![a-z0-9])(?:mot|one|ca|whole|nguyen)\s*week(?![a-z0-9])", 7),
    )

    def _meal_type(self, query):
        normalized = _normalize_for_topic(query)
        for meal_type, phrases in self.MEAL_TYPE_PATTERNS.items():
            if _topic_phrase_matches(normalized, phrases):
                return meal_type
        return None

    def _plan_horizon_days(self, query):
        normalized = _normalize_for_topic(query)
        max_days = 1
        for pattern, multiplier in self.HORIZON_NUMBER_PATTERNS:
            for match in re.finditer(pattern, normalized):
                try:
                    days = int(match.group(1)) * multiplier
                except ValueError:
                    continue
                if days > max_days:
                    max_days = days
        if max_days == 1:
            for pattern, days in self.HORIZON_WORD_PATTERNS:
                if re.search(pattern, normalized):
                    if days > max_days:
                        max_days = days
        return min(max_days, 90)

    def _numeric_metric(self, payload, keys):
        for key in keys:
            value = (payload or {}).get(key)
            if value in (None, ""):
                continue
            if isinstance(value, (int, float)):
                return float(value)
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if match:
                return float(match.group())
        return None

    def _meal_planning_results(self, results, top_k, trace, strict=True):
        blocked_terms = [
            "babyfood", "baby food", "dressing", "drsng", "sauce",
            "not included", "dry mix", "meal kits"
        ]
        ranked = []
        for item in results:
            payload = item.get("payload") or {}
            name = str(
                payload.get("food")
                or payload.get("name")
                or payload.get("Shrt_Desc")
                or payload.get("title")
                or ""
            )
            normalized_name = self.router._normalize(name)
            if any(term in normalized_name for term in blocked_terms):
                continue

            protein = self._numeric_metric(payload, ["Protein", "protein", "Protein_(g)", "proteins_100g"])
            carbs = self._numeric_metric(payload, ["Carbohydrates", "carbohydrate", "carbs", "Carbohydrt_(g)", "carbohydrates_100g"])
            fat = self._numeric_metric(payload, ["Fat", "fat", "total_fat", "Lipid_Tot_(g)", "fat_100g"])
            calories = self._numeric_metric(payload, ["Caloric Value", "calories", "Energ_Kcal", "energy-kcal_100g"])
            if strict:
                if protein is None or carbs is None or calories is None:
                    continue
            else:
                macro_count = sum(1 for value in (protein, carbs, fat) if value is not None)
                if calories is None and macro_count == 0:
                    continue

            nutrition_score = (protein or 0) - ((carbs or 0) * 0.35) - ((fat or 0) * 0.1)
            retrieval_score = float(item.get("score") or 0)
            ranked.append((nutrition_score + retrieval_score, item))

        ranked.sort(key=lambda value: value[0], reverse=True)
        selected = [item for _, item in ranked[:top_k]]
        if selected:
            trace.add(
                "Meal planning context filtered",
                "Lọc context theo protein, carb và loại bỏ payload nhiễu trước khi sinh câu trả lời.",
                evidence=[
                    str((item.get("payload") or {}).get("food") or (item.get("payload") or {}).get("name") or (item.get("payload") or {}).get("Shrt_Desc"))
                    for item in selected[:5]
                ]
            )
            return selected

        trace.add(
            "Meal planning context filtered",
            "Không có ứng viên đủ dữ liệu macro sau khi lọc; giữ context retrieval ban đầu để model hỏi thêm.",
            status="warning"
        )
        return results[:top_k]

    def _format_user_profile(self, user_profile):
        if not user_profile or not isinstance(user_profile, dict):
            return None
        parts = []
        field_map = [
            ("gender", "Giới tính"),
            ("age", "Tuổi"),
            ("height", "Chiều cao (cm)"),
            ("weight", "Cân nặng (kg)"),
            ("activityLevel", "Mức vận động"),
            ("dailyCalories", "Mục tiêu kcal/ngày"),
            ("targetWeight", "Cân nặng mục tiêu (kg)"),
            ("goal", "Mục tiêu"),
        ]
        for key, label in field_map:
            value = user_profile.get(key)
            if value is not None and value != "":
                parts.append(f"{label}: {value}")
        return "\n".join(parts) if parts else None

    async def run(
        self,
        query,
        top_k=6,
        intent=None,
        session_id=None,
        conversation_context=None,
        is_follow_up=None,
        user_profile=None
    ):
        trace = AgenticTrace()
        routed_intent = self.router.classify(query, forced_intent=intent)
        trace.add(
            "Agent Router",
            f"Phân loại query thành intent `{routed_intent}`.",
            evidence=[query]
        )

        profile_text = self._format_user_profile(user_profile)
        if profile_text:
            trace.add(
                "User profile",
                "Đã nhận thông tin cá nhân từ hồ sơ user để cá nhân hóa câu trả lời.",
                evidence=profile_text.split("\n")[:6]
            )

        use_cache = (
            bool(settings.AGENTIC_CACHE_ENABLED)
            and not session_id
            and not conversation_context
            and not is_follow_up
        )
        cache_key = self._cache_key(
            query=query,
            top_k=top_k,
            intent=routed_intent,
            session_id=session_id,
            conversation_context=conversation_context
        )
        cached = self._cache_get(cache_key) if use_cache else None
        if cached:
            cached_trace = [
                {
                    "step": 1,
                    "title": "Agent Router",
                    "text": f"Phân loại query thành intent `{routed_intent}`.",
                    "status": "done",
                    "evidence": [query],
                    "detail": None
                },
                {
                    "step": 2,
                    "title": "Redis agentic cache",
                    "text": "Trả kết quả từ cache theo session/query/context digest.",
                    "status": "done",
                    "evidence": [f"session_id={session_id}"] if session_id else [],
                    "detail": "Cache chỉ dùng cho truy vấn lặp lại cùng ngữ cảnh để tiết kiệm Redis và giảm độ trễ."
                }
            ]
            cached["trace"] = cached_trace
            cached["cache_hit"] = True
            cached.setdefault("results", [])
            return cached

        if conversation_context and self._is_pure_affirmation(query):
            anchor = self._last_assistant_question(conversation_context)
            if anchor:
                trace.add(
                    "Affirmation detected",
                    "Câu trả lời ngắn dạng đồng ý — gắn vào câu hỏi cuối của assistant để retrieval có ngữ cảnh.",
                    evidence=[query, f"anchor={anchor[:120]}"],
                    detail="Tránh trường hợp model bỏ context khi user chỉ trả lời 'có'."
                )
                query = f"{anchor.rstrip('?').strip()} - user xác nhận đồng ý ({query})"

        if self._is_vague_followup(query, is_follow_up, conversation_context):
            recent_topic = self._extract_recent_topic(conversation_context)
            topic_hint = recent_topic[:160] if recent_topic else "chủ đề vừa rồi"
            trace.add(
                "Vague follow-up detected",
                "Câu hỏi quá ngắn/chung để truy xuất chính xác nên agent xin user nói rõ thay vì gợi ý món ngẫu nhiên.",
                evidence=[query, f"topic_hint={topic_hint}"],
                detail="Tránh hallucination khi follow-up không nêu món/chủ đề cụ thể."
            )
            clarification = (
                f"Bạn muốn mình gợi ý cụ thể về điều gì liên quan đến \"{topic_hint}\"?\n\n"
                "Ví dụ:\n"
                "- Các thực phẩm tương tự (dinh dưỡng/calorie gần giống)\n"
                "- Cách kết hợp vào bữa ăn (sáng/trưa/tối)\n"
                "- Khẩu phần phù hợp với mục tiêu (giảm cân, tăng cơ...)\n\n"
                "Bạn cho mình biết hướng nào để mình tra số liệu chính xác nhé."
            )
            return {
                "type": "agentic_rag",
                "intent": routed_intent,
                "answer": clarification,
                "results": [],
                "citations": [],
                "context_used": [],
                "trace": trace.steps,
                "session_id": session_id,
                "cache_hit": False
            }

        retrieval_query = self._retrieval_query(
            query,
            conversation_context=conversation_context,
            is_follow_up=is_follow_up
        )
        if conversation_context:
            trace.add(
                "Conversation memory",
                "Đã nhận ngữ cảnh hội thoại từ backend và chỉ dùng để hiểu câu hỏi nối tiếp.",
                evidence=[f"session_id={session_id}"] if session_id else [],
                detail="Retrieval query được mở rộng bằng history khi câu hỏi hiện tại là follow-up."
            )

        if routed_intent == "image_retrieval":
            results = self.recipe_agent.image_retrieval(retrieval_query, top_k, trace)
        elif routed_intent == "recipe_reasoning":
            results = self.recipe_agent.recipe_reasoning(retrieval_query, top_k, trace)
        elif routed_intent == "ingredient_comparison":
            results = self._generic_agent().ingredient_comparison(retrieval_query, top_k, trace)
        elif routed_intent == "multi_hop":
            results = self.recipe_agent.multi_hop(retrieval_query, top_k, trace)
        elif routed_intent == "meal_planning":
            meal_type = self._meal_type(query)
            keyword_seed = self.MEAL_TYPE_SEEDS.get(meal_type, self.GENERIC_MEAL_SEED)
            trace.add(
                "Meal planning intent",
                f"Câu hỏi là lập kế hoạch bữa ăn ({meal_type or 'tổng quát'}) nên agent truy xuất dữ liệu dinh dưỡng liên quan trước khi sinh câu trả lời.",
                evidence=[f"meal_type={meal_type or 'unspecified'}"],
                detail="Query expansion chỉ seed từ khóa của bữa được hỏi để tránh kéo món sai bữa vào top-K."
            )
            generic_agent = self._generic_agent()
            meal_query = f"{retrieval_query}\n{keyword_seed}"
            results = generic_agent.run(
                meal_query,
                top_k,
                trace,
                collections=generic_agent._nutrition_collections()
            )
            results = self._meal_planning_results(results, top_k, trace)
        elif routed_intent == "weight_projection":
            trace.add(
                "Weight projection intent",
                "Câu hỏi là ước tính tăng/giảm cân nên response generator sẽ dùng công thức năng lượng và nêu giả định.",
                detail="Không truy xuất món ăn ngẫu nhiên; prompt Agentic RAG yêu cầu dùng chênh lệch kcal / 7700 khi có đủ dữ liệu."
            )
            results = []
        elif routed_intent == "nutrition_qa":
            generic_agent = self._generic_agent()
            results = generic_agent.run(
                retrieval_query,
                top_k,
                trace,
                collections=generic_agent._focused_nutrition_collections(retrieval_query)
            )
        else:
            results = self._generic_agent().run(retrieval_query, top_k, trace)

        context = self._context_from_results(results)
        citations = CitationBuilder.dedupe([
            result.get("citation") or CitationBuilder.from_payload(result.get("payload"))
            for result in results
        ])

        if not context and routed_intent not in ("weight_projection",):
            trace.add(
                "Response Generator",
                "Không tìm thấy dữ liệu phù hợp trong Qdrant. Trả lời thân thiện và gợi ý user mô tả cụ thể hơn.",
                status="warning"
            )
        else:
            trace.add(
                "Response Generator",
                "Sinh câu trả lời cuối cùng từ context đã truy xuất và citation.",
                evidence=[
                    str(citation.get("title") or citation.get("dataset") or citation.get("collection"))
                    for citation in citations[:5]
                ]
            )
        answer = await self.response_generator.generate(
            query=query,
            intent=routed_intent,
            context=context,
            citations=citations,
            trace=trace.steps,
            conversation_context=conversation_context,
            user_profile_text=profile_text
        )

        response = {
            "type": "agentic_rag",
            "intent": routed_intent,
            "answer": answer,
            "results": results,
            "citations": citations,
            "context_used": context[:5],
            "trace": trace.steps,
            "session_id": session_id,
            "cache_hit": False
        }
        if use_cache:
            self._cache_set(cache_key, response)
        return response
