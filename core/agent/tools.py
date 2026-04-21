import pandas as pd
from core.services.search_service import SearchService
from core.services.chart_service import ChartService

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

    # ================= COMPUTE =================
    def run_compute(self, df, op):

        if df is None or df.empty:
            return df

        op = op.lower()

        if "compare" in op or "so sánh" in op:
            return df.head(2)

        if "top" in op:
            col = "protein" if "protein" in op else "calories"
            if col in df.columns:
                return df.sort_values(col, ascending=False).head(10)

        if "average" in op or "trung bình" in op:
            return pd.DataFrame([df.mean(numeric_only=True)])

        return df

    # ================= CHART =================
    def run_chart(self, df, chart_type):

        if df is None or df.empty:
            return None

        return self.chart.bar_auto(df)