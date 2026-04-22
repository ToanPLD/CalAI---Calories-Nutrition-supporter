from datetime import datetime

class UserTrackingService:

    def __init__(self):
        self.storage = {}  # 👉 production dùng Redis / Postgres

    def log_meal(self, user_id, result):

        if user_id not in self.storage:
            self.storage[user_id] = []

        self.storage[user_id].append({
            "time": datetime.now().isoformat(),
            "dish": result["dish_name"],
            "calories": result["nutrition"]["calories"]
        })

    def get_daily_calories(self, user_id):

        if user_id not in self.storage:
            return 0

        today = datetime.now().date()

        return sum(
            x["calories"]
            for x in self.storage[user_id]
            if datetime.fromisoformat(x["time"]).date() == today
        )