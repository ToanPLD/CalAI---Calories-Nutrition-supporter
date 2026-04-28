from core.services.search_service import SearchService
from core.features.chart_service import ChartService
from core.services.intent_service import IntentService
from core.services.analytics_service import AnalyticsService


class QueryPipeline:

    def __init__(self):
        self.search = SearchService()
        self.chart = ChartService()
        self.intent = IntentService()
        self.analytics = AnalyticsService()

    def run(self, query):

        intent = self.intent.detect(query)
        q = query.lower()

        df = self.search.search(query)

        if df.empty:
            return {"type": "text", "data": []}

        # ================= TOP =================
        if intent == "top_n":

            col = "protein" if "protein" in q else "calories"
            if col not in df.columns:
                col = "final_score" if "final_score" in df.columns else df.columns[0]

            df2 = self.analytics.top_n(df, col)

            x_col = "food_name" if "food_name" in df2.columns else df2.columns[0]
            chart = self.chart.bar(df2, x_col, col, f"Top {col}")

            return {
                "type": "chart",
                "chart_path": chart,
                "data": df2.to_dict("records")
            }

        # ================= COMPARE =================
        if intent == "compare":

            col = "calories"
            if col not in df.columns:
                col = "final_score" if "final_score" in df.columns else df.columns[0]

            df2 = self.analytics.compare(df, col)

            x_col = "food_name" if "food_name" in df2.columns else df2.columns[0]
            chart = self.chart.bar(df2, x_col, col, "Compare")

            return {
                "type": "chart",
                "chart_path": chart,
                "data": df2.to_dict("records")
            }

        # ================= PIE =================
        if intent == "pie":

            pie_col = "category" if "category" in df.columns else df.columns[0]
            chart = self.chart.pie(df, pie_col, "Distribution")

            return {
                "type": "chart",
                "chart_path": chart,
                "data": []
            }

        # ================= DEFAULT =================
        return {
            "type": "text",
            "data": df.head(10).to_dict("records")
        }
