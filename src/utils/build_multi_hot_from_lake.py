"""
utils.build_multi_hot_from_lake

Đọc toàn bộ Parquet trong Data Lake (partition by year/month/day),
hợp nhất thành lịch sử dữ liệu và xây lại multi_hot_matrix.csv.

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
        try:
            df = pd.read_parquet(f)
            # Chỉ thêm nếu file có dữ liệu
            if not df.empty:
                dfs.append(df)
        except Exception as e:

            print(f"Loi doc file {f}: {e}")

    if not dfs:
        print("No data loaded from parquet files.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Sắp xếp nếu có cột thời gian
    if "event_time" in full_df.columns:
        full_df = full_df.sort_values(by="event_time")
    
    full_df.reset_index(drop=True, inplace=True)

    print("Loaded total rows:", len(full_df))
    return full_df


def extract_multi_hot(df):
    if df.empty:
        print("Warning: DataFrame rong, khong the trich xuat multi-hot.")
        return pd.DataFrame(columns=["draw_id", "date"] + [f"n_{i}" for i in range(1, N_POS + 1)])

    # Chuẩn hóa tên cột
    df = df.rename(columns=lambda x: x.strip().lower())

    # Map cột ngày
    for cand in ["date", "ngay", "time", "draw_date", "created_at", "event_time"]:
        if cand in df.columns:
            df["date"] = pd.to_datetime(df[cand], errors='coerce')
            break
    else:
        print("Warning: Khong tim thay cot ngay, dang de NaT")
        df["date"] = pd.NaT

    # Map cột draw_id
    found_id = False
    for cand in ["draw_id", "id", "draw", "ky"]:
        if cand in df.columns:
            # Bỏ các dòng mà draw_id bị Null/NaN trước khi convert sang int
            df = df.dropna(subset=[cand])
            df["draw_id"] = df[cand].astype(int)
            found_id = True
            break
    
    if not found_id:
        raise KeyError("Khong tim thay cot draw_id trong parquet (da thu: draw_id, id, draw, ky).")

    # Map các cột số
    # Trường hợp 1: Dữ liệu từ luồng stream (có cột 'top_numbers' là list)
    if "top_numbers" in df.columns:
        def parse_numbers(x):
            if isinstance(x, (list, np.ndarray)):
                return list(x)[:N_POS]
            return [0]*N_POS

        nums = df["top_numbers"].apply(parse_numbers).tolist()
        num_df = pd.DataFrame(nums, columns=[f"n_{i}" for i in range(1, N_POS + 1)], index=df.index)
        df = pd.concat([df, num_df], axis=1)

    number_cols = sorted([c for c in df.columns if c.startswith("n_") and c[2:].isdigit()])

    # Nếu vẫn không đủ cột, thử tìm cột 'numbers' (array)
    if len(number_cols) < N_POS and "numbers" in df.columns:
         def parse_numbers_generic(x):
            if hasattr(x, 'tolist'): return x.tolist()[:N_POS]
            if isinstance(x, list): return x[:N_POS]
            return [0]*N_POS
            
         nums = df["numbers"].apply(parse_numbers_generic).tolist()
         num_cols_new = [f"n_{i}" for i in range(1, N_POS + 1)]
         num_df = pd.DataFrame(nums, columns=num_cols_new, index=df.index)
         df = pd.concat([df, num_df], axis=1)
         number_cols = num_cols_new

    if len(number_cols) != N_POS:
        print(f"Warning: Khong tim thay dung {N_POS} cot so. Thuc te: {number_cols}. Dang bo qua cac dong loi.")
        return pd.DataFrame()

    out_df = df[["draw_id", "date"] + number_cols].copy()
    return out_df

def main():
    print("Building multi_hot_matrix.csv from Data Lake...")

    df = load_all_data()

    out_df = extract_multi_hot(df)

    if out_df.empty:
        print("Ket qua rong, khong ghi file.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_CSV, index=False, sep=";")
    print(f"DONE: Saved {OUTPUT_CSV} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()