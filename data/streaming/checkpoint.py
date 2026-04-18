# data/streaming/checkpoint.py

import json
from pathlib import Path

class CheckpointManager:

    def __init__(self, path="data/checkpoint.json"):
        self.path = Path(path)

    def save(self, index: int):
        self.path.write_text(json.dumps({"index": index}))

    def load(self):
        if not self.path.exists():
            return 0
        return json.loads(self.path.read_text()).get("index", 0)