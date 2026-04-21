import pandas as pd

def analyze(results):

    data = [r.payload for r in results]

    df = pd.DataFrame(data)

    print("\n📊 Stats:")
    print(df.describe())

    return df