import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import uuid


class ChartService:

    def bar(self, df, x, y, title):

        if x not in df.columns or y not in df.columns:
            print("❌ Missing column for chart")
            print("Columns:", df.columns)
            return None

        plt.figure(figsize=(8, 5))
        plt.bar(df[x], df[y])
        plt.title(title)
        plt.xticks(rotation=30)

        os.makedirs("charts", exist_ok=True)

        filename = f"charts/{uuid.uuid4()}.png"
        plt.savefig(filename)
        plt.close()

        return filename

    def pie(self, df, col, title):

        if col not in df.columns:
            return None

        plt.figure()
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        plt.title(title)

        filename = f"charts/{uuid.uuid4()}.png"
        plt.savefig(filename)
        plt.close()

        return filename