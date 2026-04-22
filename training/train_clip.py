# # training/train_clip.py

# import torch
# from transformers import CLIPProcessor, CLIPModel
# from torch.utils.data import DataLoader
# from clip_dataset import FoodCLIPDataset
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model.to(device)
# model.train()

# dataset = FoodCLIPDataset("data/storage/train_clip.jsonl")

# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# for epoch in range(3):
#     for batch in loader:

#         inputs = processor(
#             text=batch["text"],
#             images=batch["image"],
#             return_tensors="pt",
#             padding=True
#         ).to(device)

#         outputs = model(**inputs)

#         image_embeds = outputs.image_embeds
#         text_embeds = outputs.text_embeds

#         # 🔥 normalize
#         image_embeds = F.normalize(image_embeds, dim=1)
#         text_embeds = F.normalize(text_embeds, dim=1)

#         # 🔥 contrastive loss
#         logits = image_embeds @ text_embeds.T
#         labels = torch.arange(len(logits)).to(device)

#         loss_i = F.cross_entropy(logits, labels)
#         loss_t = F.cross_entropy(logits.T, labels)

#         loss = (loss_i + loss_t) / 2

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch} Loss: {loss.item()}")

# # save
# model.save_pretrained("models/clip-food")
# processor.save_pretrained("models/clip-food")