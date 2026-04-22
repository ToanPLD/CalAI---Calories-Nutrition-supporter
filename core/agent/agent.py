from core.agent.planner import Planner
from core.agent.executor import Executor
from core.agent.tools import AgentTools
from core.services.explain_service import ExplainService
from core.services.plan_service import PlanService


class DataAgent:

    def __init__(self):
        self.planner = Planner()
        self.tools = AgentTools()
        self.executor = Executor(self.tools)

        self.explainer = ExplainService()
        self.plan_service = PlanService()

    def detect_intent(self, query):

        q = query.lower()

        if any(k in q for k in [
            "lịch", "plan", "kế hoạch", "thực đơn",
            "diet", "tăng cân", "giảm cân"
        ]):
            return "plan"

        return "search"

    def run(self, query):

        intent = self.detect_intent(query)

        if intent == "plan":
            return self.plan_service.generate(query)

        plan = self.planner.plan(query)
        df, chart = self.executor.execute(plan)
        explanation = self.explainer.explain(df, query)

        return {
            "type": "data",
            "data": df,
            "chart": chart,
            "explanation": explanation
        }