import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import uuid


class ChartService:

    def __init__(self):
        self.dir = "charts"
        os.makedirs(self.dir, exist_ok=True)

    def bar(self, df, x, y, title):
        plt.figure()
        plt.bar(df[x], df[y])
        plt.xticks(rotation=45)
        plt.title(title)

        file = f"{uuid.uuid4()}.png"
        path = os.path.join(self.dir, file)

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    def pie(self, df, col, title):
        plt.figure()
        df[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(title)

        file = f"{uuid.uuid4()}.png"
        path = os.path.join(self.dir, file)

        plt.savefig(path)
        plt.close()

        return path