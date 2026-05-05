import math
import re
import unicodedata

import numpy as np

from config.settings import settings
from core.embedding.clip_service import CLIPService


class ViTCNNFoodClassifier:
    """
    CLIP uses a ViT image encoder, so this service gives the image pipeline
    a concrete ViT-based visual classifier without adding another heavy model.
    """

    CANDIDATES = [
        {
            "label": "bun bo hue",
            "display": "bún bò Huế",
            "aliases": ["bun bo hue", "beef noodle soup", "vietnamese spicy beef noodle soup"],
            "ingredients": ["bún", "thịt bò", "nước dùng", "rau thơm"],
            "visual_form": "noodle_soup",
            "category": "main",
        },
        {
            "label": "bun rieu",
            "display": "bún riêu",
            "aliases": ["bun rieu", "crab noodle soup", "tomato crab noodle soup"],
            "ingredients": ["bún", "cà chua", "riêu cua", "nước dùng"],
            "visual_form": "noodle_soup",
            "category": "main",
        },
        {
            "label": "pho",
            "display": "phở",
            "aliases": ["pho", "vietnamese noodle soup", "beef pho", "chicken pho"],
            "ingredients": ["bánh phở", "nước dùng", "thịt", "rau thơm"],
            "visual_form": "noodle_soup",
            "category": "main",
        },
        {
            "label": "com tam",
            "display": "cơm tấm",
            "aliases": ["com tam", "broken rice", "vietnamese rice plate"],
            "ingredients": ["cơm", "sườn", "trứng", "đồ chua"],
            "visual_form": "rice_plate",
            "category": "main",
        },
        {
            "label": "pizza",
            "display": "pizza",
            "aliases": ["pizza", "cheese pizza", "pepperoni pizza"],
            "ingredients": ["đế bánh", "phô mai", "sốt cà chua"],
            "visual_form": "pizza",
            "category": "main",
        },
        {
            "label": "sushi",
            "display": "sushi",
            "aliases": ["sushi", "maki", "nigiri", "sashimi", "sushi platter"],
            "ingredients": ["cơm sushi", "rong biển", "cá", "hải sản"],
            "visual_form": "sushi",
            "category": "main",
        },
        {
            "label": "salad with chicken",
            "display": "salad gà",
            "aliases": ["salad", "chicken salad", "vegetable salad with chicken"],
            "ingredients": ["rau", "thịt gà"],
            "visual_form": "salad",
            "category": "main",
        },
        {
            "label": "rice bowl",
            "display": "cơm tô",
            "aliases": ["rice bowl", "rice plate", "cooked rice"],
            "ingredients": ["cơm", "món mặn"],
            "visual_form": "bowl",
            "category": "main",
        },
        {
            "label": "noodle soup",
            "display": "mì/bún nước",
            "aliases": ["noodle soup", "ramen", "udon", "noodles in broth"],
            "ingredients": ["mì hoặc bún", "nước dùng"],
            "visual_form": "noodle_soup",
            "category": "main",
        },
        {
            "label": "burger",
            "display": "burger",
            "aliases": ["burger", "hamburger", "cheeseburger"],
            "ingredients": ["bánh mì", "thịt", "rau", "sốt"],
            "visual_form": "sandwich",
            "category": "main",
        },
        {
            "label": "sandwich",
            "display": "sandwich",
            "aliases": ["sandwich", "banh mi", "bread sandwich"],
            "ingredients": ["bánh mì", "nhân"],
            "visual_form": "sandwich",
            "category": "main",
        },
        {
            "label": "pasta",
            "display": "pasta",
            "aliases": ["pasta", "spaghetti", "macaroni"],
            "ingredients": ["mì pasta", "sốt"],
            "visual_form": "plate",
            "category": "main",
        },
        {
            "label": "fried chicken",
            "display": "gà chiên",
            "aliases": ["fried chicken", "chicken nuggets", "crispy chicken"],
            "ingredients": ["thịt gà", "lớp bột chiên"],
            "visual_form": "plate",
            "category": "main",
        },
        {
            "label": "steak",
            "display": "bít tết",
            "aliases": ["steak", "beef steak", "grilled beef"],
            "ingredients": ["thịt bò"],
            "visual_form": "plate",
            "category": "main",
        },
        {
            "label": "soup",
            "display": "súp/canh",
            "aliases": ["soup", "broth", "stew"],
            "ingredients": ["nước dùng"],
            "visual_form": "soup",
            "category": "main",
        },
        {
            "label": "dessert",
            "display": "món tráng miệng",
            "aliases": ["dessert", "cake", "sweet", "ice cream"],
            "ingredients": ["đường", "bột hoặc sữa"],
            "visual_form": "dessert",
            "category": "dessert",
        },
    ]

    def __init__(self, clip=None):
        self.clip = clip or CLIPService()
        self._cnn = None
        self._cnn_preprocess = None
        self._cnn_categories = []
        self._cnn_device = None
        self._cnn_error = None

    def _normalize_text(self, text):
        text = unicodedata.normalize("NFKD", str(text or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.replace("đ", "d").lower()

    def _normalize_vector(self, vector):
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    def _filename_bonus(self, candidate, filename_hint):
        normalized = self._normalize_text(filename_hint)
        if not normalized:
            return 0.0

        compact = re.sub(r"[^a-z0-9]+", " ", normalized)
        aliases = [candidate["label"], candidate["display"], *candidate.get("aliases", [])]
        for alias in aliases:
            alias_norm = self._normalize_text(alias)
            alias_tokens = [token for token in re.findall(r"[a-z0-9]+", alias_norm) if len(token) > 1]
            if alias_norm and alias_norm in compact:
                return 0.18
            if alias_tokens and all(token in compact for token in alias_tokens[:3]):
                return 0.12
        return 0.0

    def _load_cnn(self):
        if not settings.IMAGE_CLASSIFIER_CNN_ENABLED:
            return False
        if self._cnn is not None:
            return True
        if self._cnn_error:
            return False

        try:
            import torch
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.IMAGENET1K_V1
            self._cnn_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cnn_preprocess = weights.transforms()
            self._cnn_categories = weights.meta.get("categories", [])
            self._cnn = resnet50(weights=weights).to(self._cnn_device)
            self._cnn.eval()
            return True
        except Exception as exc:
            self._cnn_error = str(exc)
            return False

    def _classify_cnn(self, image, top_k=5):
        if not self._load_cnn():
            return {
                "enabled": False,
                "model": settings.IMAGE_CLASSIFIER_CNN_MODEL,
                "error": self._cnn_error,
                "top_predictions": [],
            }

        try:
            import torch

            tensor = self._cnn_preprocess(image.convert("RGB")).unsqueeze(0).to(self._cnn_device)
            with torch.no_grad():
                logits = self._cnn(tensor)[0]
                probs = torch.nn.functional.softmax(logits, dim=0)
                values, indexes = torch.topk(probs, k=min(top_k, len(probs)))

            predictions = []
            for value, index in zip(values.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
                label = self._cnn_categories[index] if index < len(self._cnn_categories) else str(index)
                predictions.append({
                    "label": label,
                    "probability": round(float(value), 4),
                })

            return {
                "enabled": True,
                "model": settings.IMAGE_CLASSIFIER_CNN_MODEL,
                "top_predictions": predictions,
            }
        except Exception as exc:
            return {
                "enabled": False,
                "model": settings.IMAGE_CLASSIFIER_CNN_MODEL,
                "error": str(exc),
                "top_predictions": [],
            }

    def _cnn_bonus(self, candidate, cnn_analysis):
        labels = " ".join(
            self._normalize_text(item.get("label", ""))
            for item in (cnn_analysis or {}).get("top_predictions", [])
        )
        if not labels:
            return 0.0

        candidate_terms = [candidate["label"], candidate["display"], *candidate.get("aliases", [])]
        for term in candidate_terms:
            term_norm = self._normalize_text(term)
            if term_norm and term_norm in labels:
                return 0.08

        visual_form = candidate.get("visual_form")
        if visual_form == "pizza" and "pizza" in labels:
            return 0.10
        if visual_form == "sandwich" and any(term in labels for term in ["burger", "cheeseburger", "hotdog", "sandwich"]):
            return 0.06
        if visual_form == "noodle_soup" and any(term in labels for term in ["soup", "bowl", "hot pot", "consomme"]):
            return 0.04
        if visual_form == "sushi" and "sushi" in labels:
            return 0.08
        if visual_form == "dessert" and any(term in labels for term in ["cake", "ice cream", "trifle", "custard"]):
            return 0.06
        return 0.0

    def classify(self, image, filename_hint=None, top_k=None):
        top_k = top_k or settings.IMAGE_CLASSIFIER_TOP_K
        image_vec = self._normalize_vector(self.clip.embed_image_pil(image))
        cnn_analysis = self._classify_cnn(image)
        prompts = [
            f"a clear food photo of {candidate['label']}"
            for candidate in self.CANDIDATES
        ]
        text_vectors = [
            self._normalize_vector(vector)
            for vector in self.clip.embed_text_batch(prompts)
        ]

        raw_scores = []
        for candidate, text_vec in zip(self.CANDIDATES, text_vectors):
            score = float(np.dot(image_vec, text_vec))
            score += self._filename_bonus(candidate, filename_hint)
            score += self._cnn_bonus(candidate, cnn_analysis)
            raw_scores.append(score)

        max_score = max(raw_scores) if raw_scores else 0.0
        exp_scores = [math.exp((score - max_score) * 12) for score in raw_scores]
        total = sum(exp_scores) or 1.0

        predictions = []
        for candidate, score, exp_score in zip(self.CANDIDATES, raw_scores, exp_scores):
            predictions.append({
                "name": candidate["display"],
                "label": candidate["label"],
                "probability": round(exp_score / total, 4),
                "score": round(score, 4),
                "visual_form": candidate["visual_form"],
                "category": candidate["category"],
                "ingredients": candidate["ingredients"],
                "aliases": candidate["aliases"],
            })

        predictions.sort(key=lambda item: (item["probability"], item["score"]), reverse=True)
        return {
            "model": settings.IMAGE_CLASSIFIER_MODEL,
            "backbone": settings.IMAGE_CLASSIFIER_BACKBONE,
            "cnn_analysis": cnn_analysis,
            "top_predictions": predictions[:top_k],
            "confidence": predictions[0]["probability"] if predictions else 0,
        }

    def to_vision_seed(self, classification):
        predictions = classification.get("top_predictions") or []
        if not predictions:
            return {
                "dish_name": "unknown",
                "confidence": 0,
                "possible_dishes": [],
            }

        top = predictions[0]
        return {
            "dish_name": top["name"],
            "possible_dishes": [
                {
                    "name": item["name"],
                    "probability": item["probability"],
                    "why": "ViT image classifier similarity",
                }
                for item in predictions[:5]
            ],
            "description": f"ViT classifier nhận diện ảnh giống {top['name']} nhất.",
            "image_observations": [
                f"ViT/CNN classifier top-1: {top['name']} ({round(top['probability'] * 100)}%)."
            ],
            "visible_vs_inferred": {
                "visible": [],
                "inferred": [top["name"], *top.get("ingredients", [])],
                "not_visible": ["khẩu phần chính xác", "gia vị ẩn", "cách nấu chi tiết"],
            },
            "identification_evidence": [
                "Nhận diện bằng mô hình ViT/CNN trên ảnh.",
            ],
            "ingredients": top.get("ingredients", []),
            "category": top.get("category") or "unknown",
            "visual_form": top.get("visual_form") or "unknown",
            "portion_description": None,
            "portion_estimation": {
                "servings": None,
                "estimated_grams": None,
                "volume_or_count": None,
                "method": "unknown",
                "uncertainty": "high",
            },
            "nutrition_estimate": {
                "calories": None,
                "protein": None,
                "carbs": None,
                "fat": None,
                "fiber": None,
                "sugar": None,
                "sodium_mg": None,
                "basis": "ViT/CNN classifier chỉ nhận diện món, không đủ dữ liệu khẩu phần để tính nutrition.",
                "main_calorie_drivers": [],
            },
            "dietary_assessment": {
                "health_score_0_10": None,
                "strengths": [],
                "concerns": [],
                "suitable_for": [],
                "caution_for": [],
            },
            "risk_flags": [],
            "recommendations": {},
            "table_rows": [],
            "uncertainty": {
                "level": "high",
                "reasons": ["Chỉ có nhận diện ảnh từ classifier, chưa có khẩu phần rõ."],
                "needs_user_input": ["Khẩu phần hoặc kích thước bát/đĩa khoảng bao nhiêu?"],
            },
            "confidence": top["probability"],
            "vit_cnn_analysis": classification,
        }
