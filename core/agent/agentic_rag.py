import asyncio
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

        if self._has_phrase(q, [
            "plan my lunch", "plan lunch", "plan my dinner", "plan dinner",
            "plan breakfast", "meal plan", "lunch plan", "dinner plan",
            "breakfast plan", "lap thuc don", "len thuc don", "thuc don",
            "bua trua", "bua toi", "bua sang", "an trua", "an toi",
            "an sang", "menu", "meal prep"
        ]):
            return "meal_planning"

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
    @staticmethod
    def from_payload(payload):
        payload = payload or {}
        explicit = payload.get("citation")
        if isinstance(explicit, dict):
            return explicit

        dataset = payload.get("source_dataset") or payload.get("domain")
        title = (
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
        )

        return {
            "dataset": dataset,
            "collection": payload.get("source_collection"),
            "row": payload.get("source_row") or payload.get("ref_id"),
            "title": title,
            "image_name": payload.get("image_name"),
            "image_file": payload.get("image_file")
        }

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

    def _display_name(self, item):
        return (
            item.get("title")
            or item.get("Name")
            or item.get("recipe_name")
            or item.get("name")
            or item.get("dish_name")
            or item.get("food_name")
            or item.get("product_name")
            or item.get("food")
            or item.get("Fruit")
            or item.get("Shrt_Desc")
            or "Nguồn dữ liệu"
        )

    def _ingredients_text(self, item, limit=6):
        ingredients = (
            item.get("cleaned_ingredients_list")
            or item.get("ingredients_list")
            or item.get("Ingredients")
            or item.get("ingredients")
        )
        if isinstance(ingredients, list):
            return ", ".join(str(value) for value in ingredients[:limit])
        if isinstance(ingredients, str):
            return ingredients[:260]
        return "-"

    def _metric(self, item, keys):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return "-"

    def _nutrition_rows(self, context):
        rows = []
        for index, item in enumerate(context[:5], start=1):
            rows.append(
                "| {idx} | {name} | {kcal} | {protein} | {carb} | {fat} | {fiber} |".format(
                    idx=index,
                    name=self._display_name(item),
                    kcal=self._metric(item, ["Calories (kcal)", "Energ_Kcal", "energy-kcal_100g", "Caloric Value", "calories"]),
                    protein=self._metric(item, ["Protein (g)", "Protein_(g)", "proteins_100g", "Protein", "protein"]),
                    carb=self._metric(item, ["Carbohydrates (g)", "Carbohydrt_(g)", "carbohydrates_100g", "Carbohydrates", "carbs"]),
                    fat=self._metric(item, ["Lipid_Tot_(g)", "fat_100g", "Fat", "fat"]),
                    fiber=self._metric(item, ["Fiber (g)", "Fiber_TD_(g)", "fiber_100g", "Dietary Fiber", "fiber"]),
                )
            )
        return rows

    def _nutrition_answer(self, query, context, citations):
        if not context:
            return (
                "Mình chưa có dữ liệu đủ chắc để trả lời chính xác. "
                "Bạn hãy gửi tên món/thực phẩm và khẩu phần ước tính, ví dụ: `1 quả táo 180g` hoặc `1 bát cơm 150g`."
            )

        rows = self._nutrition_rows(context)
        rendered_citations = CitationBuilder.dedupe([
            CitationBuilder.from_payload(item) for item in context[:5]
        ])
        source_note = self._citation_note(rendered_citations or citations)
        return "\n".join([
            "Dưới đây là dữ liệu dinh dưỡng gần nhất mình tìm được. Các số có thể khác nhau theo khẩu phần/thương hiệu.",
            "",
            "| # | Thực phẩm | kcal | Protein (g) | Carb (g) | Fat (g) | Fiber (g) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "Nếu bạn muốn tính cho một khẩu phần cụ thể, hãy gửi khối lượng hoặc số lượng bạn ăn.",
            source_note
        ]).strip()

    def _comparison_answer(self, query, context, citations):
        if not context:
            return "Mình chưa tìm được đủ dữ liệu để so sánh. Bạn có thể nêu rõ 2-3 món hoặc nguyên liệu muốn so sánh."

        has_nutrition = any(
            self._metric(item, ["Calories (kcal)", "Energ_Kcal", "energy-kcal_100g", "Caloric Value", "calories"]) != "-"
            for item in context
        )

        if has_nutrition:
            rows = []
            for index, item in enumerate(context[:8], start=1):
                rows.append(
                    "| {idx} | {term} | {name} | {kcal} | {protein} | {carb} | {fat} | {fiber} |".format(
                        idx=index,
                        term=item.get("comparison_term") or "-",
                        name=self._display_name(item),
                        kcal=self._metric(item, ["Calories (kcal)", "Energ_Kcal", "energy-kcal_100g", "Caloric Value", "calories"]),
                        protein=self._metric(item, ["Protein (g)", "Protein_(g)", "proteins_100g", "Protein", "protein"]),
                        carb=self._metric(item, ["Carbohydrates (g)", "Carbohydrt_(g)", "carbohydrates_100g", "Carbohydrates", "carbs"]),
                        fat=self._metric(item, ["Lipid_Tot_(g)", "fat_100g", "Fat", "fat"]),
                        fiber=self._metric(item, ["Fiber (g)", "Fiber_TD_(g)", "fiber_100g", "Dietary Fiber", "fiber"]),
                    )
                )

            return "\n".join([
                "Mình tìm dữ liệu dinh dưỡng gần nhất cho từng đối tượng và giữ dạng bảng để dễ so sánh.",
                "",
                "| # | Đối tượng | Kết quả phù hợp | kcal | Protein (g) | Carb (g) | Fat (g) | Fiber (g) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                *rows,
                "",
                "Kết luận nhanh: chọn theo mục tiêu chính. Nếu cần ít năng lượng hơn, ưu tiên món có kcal thấp; nếu cần no lâu, ưu tiên protein và fiber cao hơn.",
                self._citation_note(citations)
            ]).strip()

        rows = []
        for index, item in enumerate(context[:6], start=1):
            rows.append(
                f"| {index} | {item.get('comparison_term') or '-'} | {self._display_name(item)} | {self._ingredients_text(item, limit=5)} |"
            )

        return "\n".join([
            "Mình tách từng đối tượng trong câu hỏi rồi tìm dữ liệu gần nhất để so sánh:",
            "",
            "| # | Đối tượng | Kết quả phù hợp | Nguyên liệu/ghi chú chính |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "Kết luận nhanh: dùng bảng này như dữ liệu nền; nếu bạn muốn so sánh theo calories/macro, hãy gửi khẩu phần hoặc mục tiêu cụ thể.",
            self._citation_note(citations)
        ]).strip()

    def _meal_plan_answer(self, query, context, citations):
        kcal_match = re.search(r"(\d{3,4})\s*(?:kcal|calo|calories?)", query.lower())
        target_kcal = int(kcal_match.group(1)) if kcal_match else 600
        normalized_query = unicodedata.normalize("NFKD", str(query or ""))
        normalized_query = "".join(ch for ch in normalized_query if not unicodedata.combining(ch)).lower()
        normalized_query = normalized_query.replace("đ", "d").replace("Đ", "D")
        high_protein = bool(re.search(r"protein|high protein|nhieu protein|giau protein", normalized_query))
        protein_target = 35 if high_protein else 25

        protein_label = "ức gà/cá/đậu phụ/trứng" if high_protein else "nguồn protein nạc"
        return "\n".join([
            f"Mình sẽ lập một bữa trưa mặc định khoảng {target_kcal} kcal. Vì bạn chưa nêu mục tiêu cân nặng, dị ứng hay món muốn ăn, đây là phương án ước tính để bắt đầu.",
            "",
            "| Thành phần | Khẩu phần gợi ý | kcal ước tính | Protein ước tính | Vai trò |",
            "| --- | --- | --- | --- | --- |",
            f"| {protein_label} | 120-160g | 180-260 | {protein_target}-45g | Giữ no, bảo vệ cơ |",
            "| Cơm/gạo lứt/khoai | 120-180g chín | 160-240 | 3-6g | Năng lượng cho buổi chiều |",
            "| Rau xanh/salad | 250-350g | 50-100 | 3-6g | Chất xơ, vi chất |",
            "| Chất béo tốt | 1 thìa dầu olive hoặc 1/4 quả bơ | 80-120 | 0-2g | Hỗ trợ hấp thu vitamin |",
            "",
            f"Tổng mục tiêu: khoảng {target_kcal - 80}-{target_kcal + 80} kcal, protein khoảng {protein_target}-45g. Nếu bạn đang giảm cân, giữ phần tinh bột ở nửa dưới; nếu tập luyện nặng, tăng tinh bột thêm 50-80g chín.",
        ]).strip()

    def _recipe_answer(self, query, context, citations):
        if not context:
            return "Mình chưa tìm được công thức phù hợp. Hãy gửi tên món rõ hơn, ví dụ: `mac and cheese`, `chicken salad`, hoặc `bún bò Huế`."

        rows = []
        for index, item in enumerate(context[:5], start=1):
            instructions = item.get("instructions") or item.get("Directions") or item.get("directions") or "-"
            rows.append(
                f"| {index} | {self._display_name(item)} | {self._ingredients_text(item)} | {str(instructions)[:180]} |"
            )

        return "\n".join([
            "Mình tìm được các công thức phù hợp nhất trong dataset:",
            "",
            "| # | Công thức | Nguyên liệu chính | Cách làm tóm tắt |",
            "| --- | --- | --- | --- |",
            *rows,
            self._citation_note(citations)
        ]).strip()

    def _image_answer(self, query, context, citations):
        if not context:
            return "Mình chưa tìm được ảnh/công thức phù hợp trong collection ảnh món ăn."

        rows = []
        for index, item in enumerate(context[:6], start=1):
            rows.append(
                f"| {index} | {self._display_name(item)} | {item.get('image_file') or item.get('image_name') or '-'} | {self._ingredients_text(item, limit=4)} |"
            )

        return "\n".join([
            "Các ảnh/công thức gần nhất theo semantic search:",
            "",
            "| # | Món | File ảnh | Nguyên liệu gợi ý |",
            "| --- | --- | --- | --- |",
            *rows,
            self._citation_note(citations)
        ]).strip()

    def _multi_hop_answer(self, query, context, citations):
        if not context:
            return "Mình chưa tìm được công thức đủ gần để gợi ý biến tấu. Hãy gửi 2-4 nguyên liệu chính bạn đang có."

        rows = []
        for index, item in enumerate(context[:5], start=1):
            rows.append(
                f"| {index} | {self._display_name(item)} | {self._ingredients_text(item, limit=6)} | {str(item.get('instructions') or item.get('Directions') or item.get('directions') or '-')[:180]} |"
            )

        return "\n".join([
            "Dựa trên nguyên liệu bạn nêu, mình tìm các công thức gần nhất rồi dùng chúng làm nền để gợi ý biến tấu:",
            "",
            "| # | Hướng biến tấu | Nguyên liệu liên quan | Cách làm/ghi chú |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "Gợi ý thực tế: chọn công thức gần nhất với nguyên liệu bạn có, giữ protein chính, đổi phần tinh bột/rau theo khẩu phần và mục tiêu calories.",
            self._citation_note(citations)
        ]).strip()

    def _citation_note(self, citations):
        if not citations:
            return ""
        return "\nNguồn/Citation: " + "; ".join(
            str(c.get("title") or c.get("dataset") or c.get("collection"))
            for c in citations[:4]
        )

    def fallback_answer(self, query, intent, context, citations):
        if intent == "meal_planning":
            return self._meal_plan_answer(query, context, citations)

        if intent == "nutrition_qa":
            return self._nutrition_answer(query, context, citations)

        if intent == "recipe_reasoning":
            return self._recipe_answer(query, context, citations)

        if intent == "image_retrieval":
            return self._image_answer(query, context, citations)

        if intent == "ingredient_comparison":
            return self._comparison_answer(query, context, citations)

        if intent == "multi_hop":
            return self._multi_hop_answer(query, context, citations)

        if not context:
            return (
                "Mình chưa tìm được dữ liệu phù hợp trong Qdrant cho câu hỏi này. "
                "Bạn có thể hỏi cụ thể hơn về món, nguyên liệu, ảnh, công thức hoặc mục tiêu dinh dưỡng."
            )

        rows = []
        has_comparison_terms = any(item.get("comparison_term") for item in context)
        for index, item in enumerate(context[:5], start=1):
            title = self._display_name(item)
            ingredients = self._ingredients_text(item)
            if has_comparison_terms:
                rows.append(
                    f"| {index} | {item.get('comparison_term') or '-'} | {title} | {ingredients or '-'} |"
                )
            else:
                rows.append(f"| {index} | {title} | {ingredients or '-'} |")

        citation_note = ""
        if citations:
            citation_note = "\n\nNguồn chính: " + "; ".join(
                str(c.get("title") or c.get("dataset") or c.get("collection"))
                for c in citations[:3]
            )

        table_header = [
            "| # | Đối tượng | Kết quả | Nguyên liệu/ghi chú |",
            "| --- | --- | --- | --- |",
        ] if has_comparison_terms else [
            "| # | Kết quả | Nguyên liệu/ghi chú |",
            "| --- | --- | --- |",
        ]

        return "\n".join([
            f"Intent đã xử lý: `{intent}`.",
            "",
            *table_header,
            *rows,
            citation_note
        ]).strip()

    async def generate(self, query, intent, context, citations, trace, conversation_context=None):
        if not context:
            return self.fallback_answer(query, intent, context, citations)

        if intent in {"ingredient_comparison", "nutrition_qa"}:
            return self.fallback_answer(query, intent, context, citations)

        agentic_question = f"""
Bạn là Response Generator trong hệ thống Agentic RAG.

Intent đã chọn: {intent}

Yêu cầu:
- Trả lời bằng tiếng Việt tự nhiên.
- Chỉ dựa vào context truy xuất được.
- Nếu là so sánh, liệt kê, số liệu, công thức nhiều bước hoặc nhiều món: dùng bảng Markdown.
- Luôn thêm phần "Nguồn/Citation" ngắn ở cuối, dùng title/dataset/row/image_name nếu có.
- Không bịa thông tin ngoài context.
- Nếu có ngữ cảnh hội thoại, chỉ dùng để hiểu đại từ/câu hỏi nối tiếp; dữ liệu trả lời vẫn phải dựa vào context truy xuất.

Ngữ cảnh hội thoại gần đây:
{conversation_context or "Không có"}

Câu hỏi user:
{query}
""".strip()

        try:
            result = await asyncio.wait_for(
                self.llm.answer_question(
                    question=agentic_question,
                    context=context
                ),
                timeout=8
            )
        except asyncio.TimeoutError:
            return self.fallback_answer(query, intent, context, citations)

        answer = result.get("answer")
        if not answer or "Không thể tạo câu trả lời" in answer:
            return self.fallback_answer(query, intent, context, citations)
        return answer


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

        configured = [
            collection for collection in settings.TEXT_COLLECTIONS
            if collection in getattr(self.qdrant, "_collections_cache", set())
        ]
        discovered = []
        for collection in getattr(self.qdrant, "_collections_cache", set()):
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
        fruit_keywords = {"apple", "banana", "orange"}
        beverage_keywords = {"milk", "coffee", "tea", "juice", "smoothie"}

        if keywords & fruit_keywords:
            preferred = [
                "food_fruit_vectors_768",
                "food_common_vectors_768",
                "food_nutrition_vectors_768",
            ]
        elif keywords & beverage_keywords:
            preferred = [
                "beverage_text_vectors_768",
                "beverage_vectors_768",
                "food_nutrition_vectors_768",
            ]
        elif keywords:
            preferred = [
                "food_common_vectors_768",
                "food_nutrition_vectors_768",
                "food_global_10k_vectors_768",
                "food_vectors_768",
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
            r"\b(?:dinh duong|nutrition|nutrient|macro|macros|calo|calorie|calories|"
            r"kcal|protein|carb|fat|fiber|chat beo|chat xo|nguyen lieu|thuc pham|"
            r"mon an|mon|food|ingredient|cac|loai|giua|nao|hon|tot hon|nen an)\b",
            " ",
            segment
        )
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
            evidence=[str((hit.payload or {}).get("name") or (hit.payload or {}).get("title") or hit.id) for hit in selected[:5]]
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
            return self.run(query, top_k, trace)

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
            str(payload.get("ingredients_search") or ""),
            str(payload.get("cleaned_ingredients") or ""),
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

    def recipe_reasoning(self, query, top_k, trace):
        ingredient = self._ingredient_hint(query)
        search_query = self._recipe_search_query(query)
        keywords = self._food_keywords(query)
        trace.add(
            "Recipe reasoning retrieval",
            "Tìm công thức bằng vector text 768 chiều trong recipe image dataset.",
            evidence=[f"search_query={search_query}"] + ([f"ingredient_filter={ingredient}"] if ingredient else [])
        )
        hits = self.rag.search_text(
            query=search_query,
            top_k=max(top_k, top_k * 4 if keywords else top_k),
            ingredient=ingredient
        )
        hits = sorted(
            hits,
            key=lambda hit: self._recipe_match_score(hit, keywords),
            reverse=True
        )[:top_k]
        results = self.rag.format_hits(hits)
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

    def _retrieval_query(self, query, conversation_context=None, is_follow_up=None):
        if not conversation_context:
            return query

        if is_follow_up is None:
            normalized = self.router._normalize(query)
            is_follow_up = len(normalized) <= 28 or self.router._has_phrase(normalized, [
                "mon nay", "mon do", "cai nay", "cai do", "no", "nay",
                "tiep", "tinh tiep", "vay con", "so sanh voi", "them",
                "bot", "doi sang", "nhu tren", "this", "that", "it"
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
        return "agentic_rag:v6:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

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

    async def run(
        self,
        query,
        top_k=6,
        intent=None,
        session_id=None,
        conversation_context=None,
        is_follow_up=None
    ):
        trace = AgenticTrace()
        routed_intent = self.router.classify(query, forced_intent=intent)
        trace.add(
            "Agent Router",
            f"Phân loại query thành intent `{routed_intent}`.",
            evidence=[query]
        )

        cache_key = self._cache_key(
            query=query,
            top_k=top_k,
            intent=routed_intent,
            session_id=session_id,
            conversation_context=conversation_context
        )
        cached = self._cache_get(cache_key)
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
        elif routed_intent in {"nutrition_qa", "meal_planning"}:
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
            conversation_context=conversation_context
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
        self._cache_set(cache_key, response)
        return response
