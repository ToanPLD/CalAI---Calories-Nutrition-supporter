class AnalyticsService:

    def top_n(self, df, column, n=10):
        return df.sort_values(column, ascending=False).head(n)

    def compare(self, df, column):
        return df.head(5)[["food_name", column]]

    def distribution(self, df, column):
        return df[[column]]