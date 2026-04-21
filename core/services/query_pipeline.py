from core.services.search_service import SearchService
from core.services.chart_service import ChartService
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

        df = self.search.search(query)

        if df.empty:
            return {"type": "text", "data": []}

        # ================= TOP =================
        if intent == "top_n":

            col = "protein" if "protein" in query else "calories"

            df2 = self.analytics.top_n(df, col)

            chart = self.chart.bar(df2, "food_name", col, f"Top {col}")

            return {
                "type": "chart",
                "chart_path": chart,
                "data": df2.to_dict("records")
            }

        # ================= COMPARE =================
        if intent == "compare":

            col = "calories"

            df2 = self.analytics.compare(df, col)

            chart = self.chart.bar(df2, "food_name", col, "Compare")

            return {
                "type": "chart",
                "chart_path": chart,
                "data": df2.to_dict("records")
            }

        # ================= PIE =================
        if intent == "pie":

            chart = self.chart.pie(df, "category", "Distribution")

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