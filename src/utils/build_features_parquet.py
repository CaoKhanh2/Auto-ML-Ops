import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from core_model.data_prep import load_multi_hot_data
from core_model.features import build_advanced_features_from_multi_hot


LAKE_DIR = Path("data/lake/features")
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

    for t in range(T):
        # --- Frequency ---
        f30 = safe_freq_window(y_all, t, 30)
        f60 = safe_freq_window(y_all, t, 60)

        # --- Recency: dùng hàm recency chuẩn ---
        # tái sử dụng build_advanced_features để đảm bảo đồng bộ
        X_full, _, _ = build_advanced_features_from_multi_hot(df, y_all)
        rec = X_full[t][-48:-3]  # 45 recency features cuối

        # --- Time ---
        date_t = pd.to_datetime(df.iloc[t]["date"])
        time_f = np.array([t, date_t.month, date_t.weekday()], dtype=float)

        feat_vec = np.concatenate([f30, f60, rec, time_f])
        all_rows.append(feat_vec)

    return np.array(all_rows)


def main():
    print("Loading multi_hot_matrix.csv ...")
    df, y_all, y_cols = load_multi_hot_data("data/multi_hot_matrix.csv")

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
