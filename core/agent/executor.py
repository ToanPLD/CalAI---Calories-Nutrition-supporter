class Executor:

    def __init__(self, tools):
        self.tools = tools

    def execute(self, plan):

        df = None
        chart_path = None

        for step in plan:

            tool = step.get("tool")
            print("👉 STEP:", step)

            if tool == "search":
                df = self.tools.run_search(step.get("query"))

            elif tool == "filter" and df is not None:
                df = self.tools.run_filter(df, step.get("condition"))

            elif tool == "compute" and df is not None:
                df = self.tools.run_compute(df, step.get("operation"))

            elif tool == "chart" and df is not None:
                chart_path = self.tools.run_chart(df, step.get("type"))

        return df, chart_path