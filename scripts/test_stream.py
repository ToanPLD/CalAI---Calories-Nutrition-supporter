from datasets import load_dataset

ds = load_dataset(
    "pinkieseb/nutrition_dataset",
    split="train",
    streaming=True
)

for i, item in enumerate(ds):
    print(item)
    if i > 3:
        break