import pandas as pd
from pathlib import Path


class BaseCSVLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self):
        return pd.read_csv(self.file_path)

    def iterate(self):
        df = self.load()
        for _, row in df.iterrows():
            yield row.to_dict()