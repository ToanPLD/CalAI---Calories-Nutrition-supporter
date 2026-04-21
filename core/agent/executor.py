class Executor:

    def __init__(self, tools):
        self.tools = tools

    def execute(self, plan):

        df = None
        chart = None

        for step in plan["steps"]:
            tool = step["tool"]

            print("👉 STEP:", step)

            if tool == "search":
                df = self.tools.run_search(step.get("query", ""))

            elif tool == "filter":
                df = self.tools.run_filter_structured(df, step.get("filters", {}))

            elif tool == "compute":
                df = self.tools.run_compute(df, step.get("compute", ""))

            elif tool == "chart":
                chart = self.tools.run_chart(df, step.get("chart", "bar"))

        return df, chart