# data/kaggle/utils.py

import os
import pandas as pd


def find_all_csv_files(dataset_path):
    csv_files = []

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    return csv_files


def load_csv_safe(file_path):
    print(f"📂 Loading: {file_path}")

    # =========================
    # TRY NORMAL READ FIRST
    # =========================
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            engine="python",
            on_bad_lines="warn"   # ⚠️ chỉ warn, không crash
        )

        print("✅ Loaded with standard parser")
    
    except Exception as e:
        print("⚠️ Standard parser failed, fallback mode...")
        print(e)

        # =========================
        # FALLBACK RAW CLEAN
        # =========================
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

        from io import StringIO
        df = pd.read_csv(
            StringIO("\n".join(fixed_lines)),
            engine="python"
        )

    # =========================
    # CLEAN DATA TYPES
    # =========================
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📈 Rows: {len(df)}")

    # remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # type-safe cleaning
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df