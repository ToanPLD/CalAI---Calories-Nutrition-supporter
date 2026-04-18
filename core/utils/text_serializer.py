# def serialize_row(payload: dict, max_fields: int = 12) -> str:
#     """
#     Smart serializer cho CLIP:
#     - Giữ field quan trọng
#     - Giảm token < 77
#     - Không mất semantic
#     """

#     priority_keys = [
#         "activity", "exercise", "sport",
#         "calories", "kcal",
#         "mets", "intensity",
#         "weight", "kg"
#     ]

#     selected = []

#     # =========================
#     # PRIORITY FIELDS
#     # =========================
#     for pk in priority_keys:
#         for key in payload:
#             if pk in key.lower():
#                 val = payload[key]
#                 if val is not None:
#                     selected.append((key, val))

#     # =========================
#     # FILL REMAINING
#     # =========================
#     for key, val in payload.items():
#         if len(selected) >= max_fields:
#             break
#         if (key, val) not in selected:
#             if val is not None:
#                 selected.append((key, val))

#     # =========================
#     # BUILD TEXT (SHORT)
#     # =========================
#     parts = []

#     for key, val in selected:
#         key_clean = key.replace("_", " ").lower()

#         # compress value
#         val_str = str(val)[:20]  # 🔥 limit length

#         parts.append(f"{key_clean}:{val_str}")

#     return " | ".join(parts)
# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     # 🔥 tránh vượt CLIP limit
#     return text[:max_chars]

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     # 🔥 CLIP limit fix
#     return text[:max_chars]

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     # 🔥 cực kỳ quan trọng cho CLIP
#     return text[:max_chars]

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     return text[:max_chars]   # 🔥 FIX CLIP 77 token
# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     return text[:max_chars]   # 🔥 bắt buộc

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     # 🔥 cực quan trọng (CLIP limit 77 tokens)
#     return text[:max_chars]

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     return text[:max_chars]   # 🔥 bắt buộc cho CLIP

# def serialize_row(payload: dict, max_chars=500):

#     parts = []

#     for k, v in payload.items():

#         if v is None:
#             continue

#         value = str(v).strip()

#         if value == "" or value.lower() == "nan":
#             continue

#         parts.append(f"{k}: {value}")

#     text = " | ".join(parts)

#     return text[:max_chars]   # 🔥 tránh lỗi CLIP
def serialize_row(payload: dict, max_chars=500):
    parts = []

    for k, v in payload.items():
        if v is None:
            continue

        val = str(v).strip()

        if val == "" or val.lower() == "nan":
            continue

        parts.append(f"{k}: {val}")

    return " | ".join(parts)[:max_chars]