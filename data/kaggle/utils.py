# data/kaggle/utils.py

import os
import pandas as pd
from io import StringIO


def find_all_csv_files(dataset_path):
    csv_files = []

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    print(f"📁 Found {len(csv_files)} CSV files")
    return csv_files

def load_csv_safe(file_path):
    print(f"📂 Loading: {file_path}")

    df = None

    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False
        )

        print("⚡ Loaded with C engine")

    except Exception as e:
        print("⚠️ C engine failed → retry with python engine")

        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip",
                sep=None
            )

            print("🐍 Loaded with python engine")

        except Exception as e2:
            print("⚠️ Python engine failed → RAW FIX MODE")

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            max_cols = max(len(line.split(",")) for line in lines)
            print(f"🔍 Max columns detected: {max_cols}")

            fixed_lines = []

            for line in lines:
                parts = line.strip().split(",")

                if len(parts) < max_cols:
                    parts += [""] * (max_cols - len(parts))
                elif len(parts) > max_cols:
                    parts = parts[:max_cols]

                fixed_lines.append(",".join(parts))

            df = pd.read_csv(
                StringIO("\n".join(fixed_lines)),
                engine="python"
            )

            print("🧯 Loaded with RAW FIX")

    if df is None:
        print("❌ Failed to load file")
        return None

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df.columns = df.columns.str.strip()

    df = df.dropna(how="all")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"📊 Columns: {list(df.columns)}")
    print(f"📈 Rows: {len(df)}")

    return df
