ALLOWED_TOOLS = ["search", "filter", "compute", "chart"]

SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ALLOWED_TOOLS
                    },
                    "query": {"type": "string"},
                    "condition": {"type": "string"},
                    "compute": {"type": "string"},
                    "chart": {"type": "string"}
                },
                "required": ["tool"]
            }
        }
    },
    "required": ["steps"]
}