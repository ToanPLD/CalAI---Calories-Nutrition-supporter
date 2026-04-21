import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


SYSTEM_PROMPT = """
You are a nutrition expert.

Convert user query into JSON:

{
  "intent": "search_food | compare | recommendation",
  "keywords": [],
  "filters": {
    "calories_max": null,
    "protein_min": null,
    "fat_max": null,
    "carb_max": null
  }
}

ONLY return JSON.
"""


def understand_query(query: str):

    prompt = SYSTEM_PROMPT + "\nUser: " + query

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_start = text.find("{")
        json_text = text[json_start:]
        return json.loads(json_text)
    except:
        return {
            "intent": "search_food",
            "keywords": [query],
            "filters": {}
        }