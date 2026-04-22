def serialize_row(payload: dict, max_chars=400):
    parts = []

    for k, v in payload.items():
        if v is None:
            continue

        val = str(v).strip()

        if val == "" or val.lower() == "nan":
            continue

        parts.append(f"{k}: {val}")

    return " | ".join(parts)[:max_chars]