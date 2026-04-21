ALLOWED_TOOLS = {"search", "filter", "compute", "chart"}

class PlanValidator:

    @staticmethod
    def validate(plan):

        if "steps" not in plan:
            return False

        clean_steps = []

        for step in plan["steps"]:

            tool = step.get("tool")

            if tool not in ALLOWED_TOOLS:
                print(f"❌ Invalid tool removed: {tool}")
                continue

            clean_steps.append(step)

        plan["steps"] = clean_steps

        return len(clean_steps) > 0