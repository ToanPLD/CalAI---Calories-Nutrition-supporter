from typing import Dict, Any


class BaseSchema:

    @staticmethod
    def build_text(payload: Dict[str, Any]) -> str:
        parts = []

        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}.{k}" if prefix else k, v)
            else:
                parts.append(f"{prefix}: {obj}")

        flatten("", payload)
        return " | ".join(parts)