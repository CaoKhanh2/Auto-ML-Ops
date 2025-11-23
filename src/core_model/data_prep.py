import pandas as pd
import numpy as np


def load_multi_hot_data(csv_path):
    df = pd.read_csv(
        csv_path,
        sep=";"
    )

    # Parse date dd/mm/yyyy
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Auto rename n1 → n_1
    rename_map = {}
    for col in df.columns:
        if col.startswith("n") and not col.startswith("n_"):
            num = col.replace("n", "")
            if num.isdigit():
                rename_map[col] = f"n_{num}"
    df = df.rename(columns=rename_map)

    # Check feature columns
    y_cols = [col for col in df.columns if col.startswith("n_")]
    if not y_cols:
        raise ValueError("Không tìm thấy cột n_x trong dữ liệu.")

    y_all = df[y_cols].values.astype(int)

    return df, y_all, y_cols
