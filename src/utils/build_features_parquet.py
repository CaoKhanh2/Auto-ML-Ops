import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Make `src/` importable when this file is executed directly
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = SRC_DIR.parent

from core_model.data_prep import load_multi_hot_data
from core_model.features import compute_recency_features


LAKE_DIR = PROJECT_ROOT / "data/lake/features"
LAKE_DIR.mkdir(parents=True, exist_ok=True)


def safe_freq_window(y_all, t, w):
    """Tránh mean của empty slice → luôn trả về mean hợp lệ."""
    if t <= 0:
        return np.zeros(y_all.shape[1])
    if t - w < 0:
        return y_all[:t].mean(axis=0)
    return y_all[t-w:t].mean(axis=0)


def build_all_features(df, y_all):
    T = len(df)
    all_rows = []

    # Tính recency một lần để tránh O(T^2)
    recency_matrix = compute_recency_features(y_all)

    for t in range(T):
        # --- Frequency ---
        f30 = safe_freq_window(y_all, t, 30)
        f60 = safe_freq_window(y_all, t, 60)

        # --- Recency ---
        rec = recency_matrix[t]

        # --- Time ---
        date_t = pd.to_datetime(df.iloc[t]["date"])
        time_f = np.array([t, date_t.month, date_t.weekday()], dtype=float)

        feat_vec = np.concatenate([f30, f60, rec, time_f])
        all_rows.append(feat_vec)

    return np.array(all_rows)


def main():
    print("Loading multi_hot_matrix.csv ...")
    df, y_all, y_cols = load_multi_hot_data(PROJECT_ROOT / "data/multi_hot_matrix.csv")

    print("Building features matrix ...")
    X = build_all_features(df, y_all)

    print("Encoding as parquet rows ...")
    out_df = pd.DataFrame({
        "draw_id": df["draw_id"],
        "date": df["date"],
        "features": X.tolist(),
        "numbers": df[y_cols].values.tolist()
    })

    OUTPUT = LAKE_DIR / "features.parquet"
    out_df.to_parquet(OUTPUT, index=False)

    print(f"✔ Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
