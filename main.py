from core.search.hybrid_search import hybrid_search
from core.ai.reranker import rerank
from core.analysis.nutrition_scoring import rank_by_health
from core.recommendation.meal_engine import recommend_meal


def run():

    print("🚀 SYSTEM READY")

    query = input("👉 Query: ")

    # ===== SEARCH =====
    results = hybrid_search(query, None)

    print(f"🔍 Found {len(results)}")

    # ===== RERANK =====
    results = rerank(query, results)

    # ===== HEALTH SCORE =====
    results = rank_by_health(results)

    # ===== MEAL =====
    meal = recommend_meal(results, goal="muscle_gain")

    print("\n🥗 Recommended meal:\n")

    for m in meal:
        print(m.payload)


if __name__ == "__main__":
    run()