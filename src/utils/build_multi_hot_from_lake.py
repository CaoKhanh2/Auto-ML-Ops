"""Build multi_hot_matrix.csv by merging all parquet files in the data lake.

Đọc toàn bộ Parquet trong Data Lake (partition by year/month/day), hợp nhất
thành lịch sử dữ liệu và xây lại multi_hot_matrix.csv.

Output: data/multi_hot_matrix.csv
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


DATA_LAKE = Path("data/lake")
OUTPUT_CSV = Path("data/multi_hot_matrix.csv")

# Số lượng số trong 1 draw (ví dụ 6 số)
N_POS = 6

def list_all_parquet():
    """Trả về danh sách toàn bộ file parquet trong data lake."""
    return list(DATA_LAKE.rglob("*.parquet"))


def load_all_data():
    parquet_files = list_all_parquet()

    if not parquet_files:
        raise FileNotFoundError("Không tìm thấy file parquet nào trong data/lake.")

    print(f"Found {len(parquet_files)} parquet files in Data Lake.")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values(by="event_time")
    full_df.reset_index(drop=True, inplace=True)

    print("📌 Loaded total rows:", len(full_df))
    return full_df


def extract_multi_hot(df):
    # Chuẩn hóa tên cột
    df = df.rename(columns=lambda x: x.strip().lower())

    # Map cột ngày
    for cand in ["date", "ngay", "time", "draw_date", "created_at"]:
        if cand in df.columns:
            df["date"] = pd.to_datetime(df[cand])
            break
    else:
        raise KeyError("Không tìm thấy cột ngày trong parquet.")

    # Map cột draw_id
    for cand in ["draw_id", "id", "draw", "ky"]:
        if cand in df.columns:
            df["draw_id"] = df[cand].astype(int)
            break
    else:
        raise KeyError("Không tìm thấy cột draw_id trong parquet.")

    # Map các cột số
    number_cols = sorted([c for c in df.columns if c.startswith("n_")])

    if len(number_cols) != 6:
        raise ValueError(f"Không tìm thấy đúng 6 cột n_1..n_6. Thực tế: {number_cols}")

    out_df = df[["draw_id", "date"] + number_cols].copy()
    return out_df

def main():
    print("Building multi_hot_matrix.csv from Data Lake...")

    df = load_all_data()

    out_df = extract_multi_hot(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"DONE → Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
