class IntentService:

    def detect(self, query: str):

        q = query.lower()

        if "top" in q and ("protein" in q or "calories" in q):
            return "top_n"

        if "compare" in q or "vs" in q:
            return "compare"

        if "distribution" in q or "ratio" in q:
            return "pie"

        return "search"