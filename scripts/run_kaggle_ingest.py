from data.kaggle.kaggle_pipeline import KaggleIngestionPipeline


if __name__ == "__main__":
    pipeline = KaggleIngestionPipeline()
    pipeline.run()