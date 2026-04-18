from pipelines.query_pipeline import QueryPipeline


pipeline = QueryPipeline()

query = "running high intensity calories > 300"

results = pipeline.run(query)

for r in results:
    print(r.payload)