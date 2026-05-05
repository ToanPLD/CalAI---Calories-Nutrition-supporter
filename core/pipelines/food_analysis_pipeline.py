from core.services.vision.qwen_vl_service import QwenVLService
from core.embedding.clip_service import CLIPService
from core.embedding.text_embedding_service import TextEmbeddingService
from core.services.rag.food_rag_service import FoodRAGService
from core.services.rerank.cross_encoder import CrossEncoderReranker
from core.services.user.user_tracking import UserTrackingService
from core.services.cache.embedding_cache import EmbeddingCache
from core.services.llm.llm_service import LLMService
from core.services.vision.vit_cnn_service import ViTCNNFoodClassifier
from config.settings import settings
import re
import unicodedata

class FoodAnalysisPipeline:

    def __init__(self):
        self._qwen = None
        self._clip = None
        self._text_embed = None
        self._rag = None
        self._rerank = None
        self._user_tracking = None
        self._cache = None
        self._llm = None
        self._image_classifier = None

    @property
    def qwen(self):
        if self._qwen is None:
            self._qwen = QwenVLService()
        return self._qwen

    @property
    def clip(self):
        if self._clip is None:
            self._clip = CLIPService()
        return self._clip

    @property
    def text_embed(self):
        if self._text_embed is None:
            self._text_embed = TextEmbeddingService()
        return self._text_embed

    @property
    def rag(self):
        if self._rag is None:
            self._rag = FoodRAGService()
        return self._rag

    @property
    def rerank(self):
        if self._rerank is None:
            self._rerank = CrossEncoderReranker()
        return self._rerank

    @property
    def user_tracking(self):
        if self._user_tracking is None:
            self._user_tracking = UserTrackingService()
        return self._user_tracking

    @property
    def cache(self):
        if self._cache is None:
            self._cache = EmbeddingCache()
        return self._cache

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMService()
        return self._llm

    @property
    def image_classifier(self):
        if self._image_classifier is None:
            self._image_classifier = ViTCNNFoodClassifier(clip=self.clip)
        return self._image_classifier

    def _image_key(self, image):
        import hashlib
        return hashlib.md5(image.tobytes()).hexdigest()

    def _normalize_text(self, text):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def _is_unknown_vision(self, vision):
        dish = self._normalize_text(vision.get("dish_name", ""))
        return not dish or dish in {"unknown", "none", "null", "khong ro"}

    def _classifier_query_text(self, classification):
        predictions = (classification or {}).get("top_predictions") or []
        terms = []
        for item in predictions[:3]:
            terms.extend([
                item.get("name"),
                item.get("label"),
                *(item.get("aliases") or [])[:3],
            ])
        return " ".join(str(term) for term in terms if term).strip()

    def _merge_classifier_with_vision(self, vision, classification):
        vision = vision if isinstance(vision, dict) else {}
        classification = classification if isinstance(classification, dict) else {}
        predictions = classification.get("top_predictions") or []
        top = predictions[0] if predictions else {}

        if not predictions:
            vision["vit_cnn_analysis"] = classification
            return vision

        classifier_confidence = self._to_float(classification.get("confidence")) or 0
        if self._is_unknown_vision(vision) and classifier_confidence >= settings.IMAGE_CLASSIFIER_MIN_CONFIDENCE:
            seeded = self.image_classifier.to_vision_seed(classification)
            seeded["vit_cnn_analysis"] = classification
            return seeded

        vision["vit_cnn_analysis"] = classification
        existing = vision.get("possible_dishes")
        existing = existing if isinstance(existing, list) else []
        classifier_dishes = [
            {
                "name": item.get("name"),
                "probability": item.get("probability"),
                "why": "ViT/CNN classifier",
            }
            for item in predictions[:3]
            if item.get("name")
        ]
        vision["possible_dishes"] = existing[:5] or classifier_dishes
        if not vision.get("identification_evidence"):
            vision["identification_evidence"] = [
                f"ViT/CNN classifier top-1: {top.get('name')}."
            ]
        return vision

    def _normalize_vision_details(self, vision):
        dish = self._normalize_text(vision.get("dish_name", ""))
        category = self._normalize_text(vision.get("category", ""))
        visual_form = self._normalize_text(vision.get("visual_form", ""))

        if "pizza" in dish or "pizza" in category:
            if not visual_form and "whole" in category:
                vision["visual_form"] = "whole pizza"
            elif not visual_form:
                vision["visual_form"] = "pizza"

            portion = self._normalize_text(vision.get("portion_description", ""))
            if "slice" in portion and vision.get("visual_form") == "whole pizza":
                vision["portion_description"] = "ước tính 1 pizza nguyên chiếc cỡ vừa"

        if "sushi" in dish or "sushi" in category:
            if not visual_form or visual_form == "slice":
                vision["visual_form"] = "sushi platter"

            if dish == "sushi" and vision.get("visual_form") == "sushi platter":
                vision["dish_name"] = "sushi platter"

            portion = self._normalize_text(vision.get("portion_description", ""))
            if not portion:
                vision["portion_description"] = "nhiều miếng sushi trên đĩa"

        return vision

    def _enrich_query(self, vision):
        dish = vision.get("dish_name", "")
        desc = vision.get("description", "")
        ingredients = " ".join(vision.get("ingredients") or [])
        classifier_text = self._classifier_query_text(vision.get("vit_cnn_analysis"))
        sub_items = " ".join(
            f"{item.get('name', '')} {item.get('count', '')} {' '.join(item.get('visible_ingredients') or [])}"
            for item in (vision.get("sub_items") or [])
            if isinstance(item, dict)
        )

        if "cơm tấm" in dish.lower():
            return (
                f"{dish}. {desc}. {ingredients}. {classifier_text}. "
                "broken rice grilled pork chop shredded pork skin egg meatloaf fried egg Vietnamese rice plate"
            )

        if "pizza" in self._normalize_text(dish):
            return (
                f"{dish}. {desc}. {ingredients}. {classifier_text}. "
                "pizza cheese sausage pepperoni ham meat tomato sauce bbq sauce whole pizza"
            )

        if "sushi" in self._normalize_text(f"{dish} {desc} {ingredients} {classifier_text} {sub_items}"):
            return (
                f"{dish}. {desc}. {ingredients}. {classifier_text}. {sub_items}. "
                "sushi platter sushi set salmon nigiri maki roll uramaki avocado cucumber nori seaweed rice tempura shrimp sashimi Japanese"
            )

        return f"{dish} {desc} {ingredients} {classifier_text} {sub_items}".strip()

    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else None

    def _nutrition_estimate_from_vision(self, vision):
        if not isinstance(vision, dict):
            return None

        raw = vision.get("nutrition_estimate")
        if not isinstance(raw, dict):
            return None
        portion_estimation = vision.get("portion_estimation")
        portion_estimation = portion_estimation if isinstance(portion_estimation, dict) else {}

        estimate = {
            "calories": self._to_float(raw.get("calories")),
            "protein": self._to_float(raw.get("protein")),
            "carbs": self._to_float(raw.get("carbs")),
            "fat": self._to_float(raw.get("fat")),
            "fiber": self._to_float(raw.get("fiber")),
            "sugar": self._to_float(raw.get("sugar")),
            "sodium_mg": self._to_float(raw.get("sodium_mg")),
            "serving_size": (
                vision.get("portion_description")
                or portion_estimation.get("method")
                or raw.get("basis")
            ),
            "note": raw.get("basis") or "ước tính trực tiếp từ model vision theo khẩu phần nhìn thấy"
        }

        macro_values = [
            estimate.get("calories"),
            estimate.get("protein"),
            estimate.get("carbs"),
            estimate.get("fat")
        ]
        return estimate if any(value not in (None, 0) for value in macro_values) else None

    def _estimated_grams_from_vision(self, vision):
        if not isinstance(vision, dict):
            return None

        portion_estimation = vision.get("portion_estimation")
        if not isinstance(portion_estimation, dict):
            return None

        grams = self._to_float(portion_estimation.get("estimated_grams"))
        return grams if grams and grams > 0 else None

    def _portion_estimate_from_rag(self, vision, retrieved):
        grams = self._estimated_grams_from_vision(vision)
        per_100g = self._payload_nutrition_per_100g(retrieved)
        if not grams or not per_100g:
            return None

        scale = grams / 100
        return {
            "calories": (
                round(per_100g["calories"] * scale)
                if per_100g.get("calories") is not None
                else None
            ),
            "protein": (
                round(per_100g["protein"] * scale, 1)
                if per_100g.get("protein") is not None
                else None
            ),
            "carbs": (
                round(per_100g["carbs"] * scale, 1)
                if per_100g.get("carbs") is not None
                else None
            ),
            "fat": (
                round(per_100g["fat"] * scale, 1)
                if per_100g.get("fat") is not None
                else None
            ),
            "serving_size": f"ước tính theo phần nhìn thấy khoảng {round(grams)} g",
            "note": "tính từ dữ liệu dinh dưỡng theo 100g và khẩu phần model vision ước lượng"
        }

    def _payload_nutrition_per_100g(self, payload):
        if not payload:
            return None

        calories = (
            payload.get("energy-kcal_100g")
            or payload.get("calories")
            or payload.get("kcal")
        )
        protein = (
            payload.get("proteins_100g")
            or payload.get("protein")
            or payload.get("proteins")
        )
        carbs = (
            payload.get("carbohydrates_100g")
            or payload.get("carbohydrate")
            or payload.get("carbs")
            or payload.get("carb")
        )
        fat = payload.get("fat_100g") or payload.get("fat") or payload.get("total_fat")

        values = {
            "calories": self._to_float(calories),
            "protein": self._to_float(protein),
            "carbs": self._to_float(carbs),
            "fat": self._to_float(fat)
        }

        return values if any(value is not None for value in values.values()) else None

    def _payload_declared_nutrition(self, payload):
        if not payload:
            return None

        text = str(payload.get("nutrition") or "")
        if not text:
            return None

        patterns = {
            "calories": r"calories?\s+(\d+(?:\.\d+)?)",
            "protein": r"protein\s+(\d+(?:\.\d+)?)\s*g",
            "carbs": r"(?:total\s+)?carbohydrate\s+(\d+(?:\.\d+)?)\s*g",
            "fat": r"(?:total\s+)?fat\s+(\d+(?:\.\d+)?)\s*g"
        }

        values = {}
        lower = text.lower()
        for key, pattern in patterns.items():
            match = re.search(pattern, lower)
            values[key] = float(match.group(1)) if match else None

        return values if any(value is not None for value in values.values()) else None

    def _nutrition_summary(self, retrieved, estimated):
        per_100g = self._payload_nutrition_per_100g(retrieved)
        declared = self._payload_declared_nutrition(retrieved)
        matched_item = (
            retrieved.get("name")
            or retrieved.get("dish_name")
            or retrieved.get("food_name")
            or retrieved.get("product_name")
            or retrieved.get("recipe_name")
            if retrieved else None
        )

        estimated = estimated or None

        if per_100g and estimated:
            note = (
                "RAG cung cấp chỉ số theo 100g; estimated_visible_portion là ước tính cho phần ăn nhìn thấy trong ảnh."
            )
        elif retrieved:
            note = (
                "RAG cung cấp recipe/context phù hợp nhưng không có calories chuẩn theo khẩu phần; "
                "estimated_visible_portion chỉ có khi model vision hoặc khẩu phần đủ dữ liệu."
            )
        else:
            note = "Không có RAG phù hợp; chỉ trả dữ liệu model vision quan sát được từ ảnh."

        return {
            "matched_item": matched_item,
            "basis": (
                retrieved.get("serving_size")
                or ("100 g" if per_100g else (estimated or {}).get("serving_size"))
                if retrieved else (estimated or {}).get("serving_size")
            ),
            "per_100g": per_100g,
            "declared_nutrition": declared,
            "estimated_visible_portion": estimated,
            "note": note
        }

    async def analyze(self, image, user_id=None, filename=None, question=None):

        # STEP 1: VIT/CNN IMAGE CLASSIFIER + VISION
        vit_cnn = {}
        if settings.IMAGE_CLASSIFIER_ENABLED:
            vit_cnn = self.image_classifier.classify(image, filename_hint=filename)

        classifier_hint = self._classifier_query_text(vit_cnn)
        filename_hint = (
            f"{filename or ''}\nViT/CNN classifier hints: {classifier_hint}"
            if classifier_hint
            else filename
        )
        vision = await self.qwen.analyze_food(image, filename_hint=filename_hint)
        # The filename is already passed to the vision model as a weak hint.
        # Do not replace the visual result with a hard-coded filename guess.

        vision = self._merge_classifier_with_vision(vision, vit_cnn)
        vision = self._normalize_vision_details(vision)

        dish = vision.get("dish_name", "")
        confidence = self._to_float(vision.get("confidence")) or 0
        if confidence > 1:
            confidence = confidence / 100
        query_text = self._enrich_query(vision)
        has_visual_dish = bool(dish and not self._is_unknown_vision(vision))
        has_classifier_hint = bool(self._classifier_query_text(vit_cnn))
        should_search = has_visual_dish or has_classifier_hint

        # STEP 2: CACHE
        img_key = "img_" + self._image_key(image)
        txt_key = "txt768_" + query_text

        image_vec = None
        text_vec = None
        if should_search:
            image_vec = self.cache.get_or_set(
                img_key,
                lambda: self.clip.embed_image_pil(image)
            )

            search_text = query_text if has_visual_dish else self._classifier_query_text(vit_cnn)
            text_vec = self.cache.get_or_set(
                "txt768_" + search_text,
                lambda: self.text_embed.embed(search_text)
            )

        # STEP 3: RAG
        hits = []
        if should_search:
            hits = self.rag.hybrid_search(
                image_vec,
                text_vec,
                dish,
                vision_context=vision,
                top_k=settings.RAG_CANDIDATE_TOP_K
            )

        # STEP 4: RERANK
        if hits:
            hits = self.rerank.rerank(query_text, hits)

        best = hits[0].payload if hits else {}

        # STEP 5: NUTRITION ESTIMATE
        vision_estimate = self._nutrition_estimate_from_vision(vision)
        rag_portion_estimate = self._portion_estimate_from_rag(vision, best)
        if vision_estimate:
            estimated, estimate_source = vision_estimate, "vision_model_estimate"
        elif rag_portion_estimate:
            estimated, estimate_source = rag_portion_estimate, "rag_portion_estimate"
        else:
            estimated, estimate_source = None, "not_available"
        summary = self._nutrition_summary(best, estimated)

        warnings = []
        if confidence < settings.VISION_MIN_CONFIDENCE:
            warnings.append(
                "Độ tin cậy nhận diện thấp; kết quả nutrition được xem là ước tính."
            )
        if best and self.rag._is_packaged_payload(best):
            warnings.append(
                "Dữ liệu RAG là sản phẩm đóng gói; chỉ nên dùng khi ảnh có bao bì/nhãn tương ứng."
            )

        # STEP 6: RESULT
        result = {
            "dish_name": dish,
            "confidence": confidence,
            "vision_detail": {
                "image_quality": vision.get("image_quality", {}),
                "description": vision.get("description"),
                "possible_dishes": vision.get("possible_dishes", []),
                "image_observations": vision.get("image_observations", []),
                "visible_vs_inferred": vision.get("visible_vs_inferred", {}),
                "identification_evidence": vision.get("identification_evidence", []),
                "ingredients": vision.get("ingredients", []),
                "sub_items": vision.get("sub_items", []),
                "category": vision.get("category"),
                "visual_form": vision.get("visual_form"),
                "portion_description": vision.get("portion_description"),
                "portion_estimation": vision.get("portion_estimation", {}),
                "health_context": vision.get("health_context", {}),
                "dietary_assessment": vision.get("dietary_assessment", {}),
                "risk_flags": vision.get("risk_flags", []),
                "recommendations": vision.get("recommendations", {}),
                "table_rows": vision.get("table_rows", []),
                "uncertainty": vision.get("uncertainty", {}),
                "vit_cnn_analysis": vision.get("vit_cnn_analysis", {})
            },
            "estimated_nutrition": estimated,
            "retrieved_nutrition": best,
            "nutrition_summary": summary,
            "nutrition_source": (
                "rag_with_portion_estimate"
                if best and estimate_source == "rag_portion_estimate"
                else "rag_with_vision_estimate"
                if best and estimated
                else ("rag" if best else estimate_source)
            ),
            "warnings": warnings,
            "analysis_note": (
                "Không dùng kết quả RAG nếu payload không liên quan trực tiếp tới món đã nhận diện."
                if not best else "Dữ liệu RAG đã vượt qua kiểm tra liên quan tới món."
            )
        }

        answer = await self.llm.answer_food_image(
            question=question or "Đây là món gì? Hãy phân tích dinh dưỡng và tư vấn.",
            analysis=result
        )
        if answer:
            result["answer"] = answer

        # STEP 7: USER TRACK
        if user_id:
            self.user_tracking.log_meal(user_id, result)

        return result
