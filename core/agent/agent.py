from core.agent.planner import Planner
from core.agent.validator import PlanValidator
from core.agent.executor import Executor
from core.agent.tools import AgentTools

class DataAgent:

    def __init__(self):
        self.planner = Planner()
        self.tools = AgentTools()
        self.executor = Executor(self.tools)

    def run(self, query):

        plan = self.planner.plan(query)

        if not PlanValidator.validate(plan):
            print("⚠️ fallback plan")

            plan = {
                "steps": [
                {"tool": "search", "query": query},
                {"tool": "compute", "compute": "top"}
        ]
    }

        df, chart = self.executor.execute(plan)

        return {
            "df": df,
            "chart": chart,
            "plan": plan
        }