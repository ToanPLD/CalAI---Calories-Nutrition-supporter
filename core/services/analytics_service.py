class AnalyticsService:

    def top_n(self, df, column, n=10):
        return df.sort_values(column, ascending=False).head(n)

    def compare(self, df, column):
        cols = []
        if "food_name" in df.columns:
            cols.append("food_name")
        cols.append(column)
        return df.head(5)[cols]

    def distribution(self, df, column):
        return df[[column]]
