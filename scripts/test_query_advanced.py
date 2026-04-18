from pipelines.query_pipeline import QueryPipeline

pipeline = QueryPipeline()

# ================= TEXT QUERY =================
results = pipeline.query(
    text="high protein low carb food",
    filters={
        "min_protein": 20,
        "max_carbs": 20
    }
)

print("\n=== RESULT ===")
for r in results:
    print(r.payload)