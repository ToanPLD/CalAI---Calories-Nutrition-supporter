# scripts/test_query.py

from pipelines.query_pipeline import QueryPipeline

if __name__ == "__main__":
    pipeline = QueryPipeline()

    image_path = "data/storage/images/0000011e-a803-4225-b691-c56ed2f4ce1b.jpg"

    result = pipeline.run(image_path)

    print("\nRESULT:")
    print(result)