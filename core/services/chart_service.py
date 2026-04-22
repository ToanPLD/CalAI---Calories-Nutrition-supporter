import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import uuid


class ChartService:

    def auto_chart(self, df):

        if df is None or df.empty:
            return None

        cols = df.columns

        if any("name" in c.lower() for c in cols):
            return self.bar_auto(df)

        if len(df) <= 5:
            col = cols[0]
            return self.pie(df, col, "Distribution")

        return self.bar_auto(df)

    def bar(self, df, x, y, title="Result"):

        if x not in df.columns or y not in df.columns:
            print("❌ Missing column")
            print(df.columns)
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

    def bar_auto(self, df):

        if df is None or df.empty:
            return None

        x = next(
            (c for c in df.columns if "name" in c.lower()),
            None
        )

        if x is None:
            x = df.columns[0]

        y = next(
            (c for c in ["calories", "protein", "carb", "fat"] if c in df.columns),
            None
        )

        if y is None:
            print("❌ No numeric column for chart")
            return None

        return self.bar(df, x, y, "Auto Chart")