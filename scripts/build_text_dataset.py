import json

input_file = "data/storage/nutrition_data.jsonl"
output_file = "data/storage/train_clip.jsonl"

with open(input_file) as f, open(output_file, "w") as out:
    for line in f:
        item = json.loads(line)

        text = f"""
        A food dish with {item['calories']} calories,
        {item['protein_g']}g protein,
        {item['carbs_g']}g carbs,
        {item['fat_g']}g fat
        """

        out.write(json.dumps({
            "image_path": item["image_path"],
            "text": text
        }) + "\n")