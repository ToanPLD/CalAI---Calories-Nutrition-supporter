from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

print("Torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Text embedding model:", settings.TEXT_EMBEDDING_MODEL)
print("Image embedding model:", settings.IMAGE_EMBEDDING_MODEL)
print("Vision model:", settings.VISION_MODEL)
print("LLM model:", settings.LLM_MODEL)
