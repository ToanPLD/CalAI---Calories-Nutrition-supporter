import torch
from transformers import ViTModel

print("Torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

model = ViTModel.from_pretrained("google/vit-base-patch16-224")
print("ViT loaded OK")