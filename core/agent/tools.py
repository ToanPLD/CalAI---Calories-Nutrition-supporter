import pandas as pd
from core.services.search_service import SearchService
from core.features.chart_service import ChartService

class AgentTools:

    def __init__(self):
        self.search = SearchService()
        self.chart = ChartService()

    # ================= SEARCH =================
    def run_search(self, query):
        return self.search.search(query)

    # ================= FILTER (SAFE) =================
    def run_filter_structured(self, df, filters):

        if df is None or df.empty:
            return df

        for col, cond in filters.items():

            if col not in df.columns:
                continue

            if "lt" in cond:
                df = df[df[col] < cond["lt"]]

            if "gt" in cond:
                df = df[df[col] > cond["gt"]]


        return df

    def run_filter(self, df, condition):
        if df is None or df.empty or not condition:
            return df

        if isinstance(condition, dict):
            return self.run_filter_structured(df, condition)

        text = str(condition).lower()

        if ("low calorie" in text or "ít calo" in text) and "calories" in df.columns:
            return df.sort_values("calories", ascending=True)

        if ("high protein" in text or "giàu protein" in text) and "protein" in df.columns:
            return df.sort_values("protein", ascending=False)

        return df

    # ================= COMPUTE =================
    def run_compute(self, df, operation):

        if df is None or df.empty:
            return df

        op = operation.lower()

        if "compare" in op or "comparison" in op or "so sánh" in op:
            return df.head(2)

        if "top" in op:
            col = "protein" if "protein" in df.columns else "calories"
            if col not in df.columns:
                col = "final_score" if "final_score" in df.columns else df.columns[0]
            return df.sort_values(col, ascending=False).head(10)

        if "low" in op or "ít calo" in op:
            if "calories" not in df.columns:
                return df.head(10)
            return df.sort_values("calories", ascending=True).head(10)

        if "average" in op:
            return df.mean(numeric_only=True).to_frame().T

        return df

    # ================= CHART =================
    def run_chart(self, df, chart_type):

        if df is None or df.empty:
            return None

        if not chart_type:
            return self.chart.auto_chart(df)

        if chart_type == "bar":
            return self.chart.bar_auto(df)

        if chart_type == "pie":
            return self.chart.pie(df, df.columns[0], "Distribution")

        return self.chart.auto_chart(df)
