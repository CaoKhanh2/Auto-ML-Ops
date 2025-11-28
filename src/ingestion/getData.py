"""
Robust Scraper với cơ chế 'Deterministic Date Calculation'.
Đảm bảo ngày tháng luôn chính xác và liên tục, không phụ thuộc vào định dạng web.
"""

import os
import sys
import pandas as pd
import requests
import re
from pathlib import Path
from datetime import timedelta, datetime

# --- CẤU HÌNH ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/multi_hot_matrix.csv"
URL_SOURCE = "https://www.minhngoc.net.vn/ket-qua-xo-so/dien-toan-vietlott/mega-6x45.html"

# --- LOGIC TÍNH TOÁN NGÀY (CORE) ---
def get_next_schedule_date(current_date):
    """
    Tính ngày quay tiếp theo dựa trên thứ trong tuần.
    Quy luật: Thứ 4 (+2) -> Thứ 6 (+2) -> CN (+3) -> Thứ 4.
    """
    wd = current_date.weekday()
    if wd == 2:   # Thứ 4 -> Thứ 6
        return current_date + timedelta(days=2)
    elif wd == 4: # Thứ 6 -> Chủ Nhật
        return current_date + timedelta(days=2)
    elif wd == 6: # Chủ Nhật -> Thứ 4 tuần sau
        return current_date + timedelta(days=3)
    else:
        # Nếu ngày gốc bị lệch (không rơi vào T4, T6, CN), tự động chỉnh về nhịp gần nhất
        # Đây là cơ chế tự sửa lỗi (Self-healing)
        return current_date + timedelta(days=1)

def calculate_future_date(last_id, last_date, target_id):
    """
    Tính toán ngày cho target_id dựa trên mốc last_id/last_date.
    Hỗ trợ trường hợp bị missed nhiều kỳ (ví dụ db đang 1437, web đã ra 1440).
    """
    curr_date = last_date
    # Lặp qua từng kỳ còn thiếu để cộng dồn ngày
    for _ in range(last_id, target_id):
        curr_date = get_next_schedule_date(curr_date)
    return curr_date

# --- SCRAPER ---
def fetch_raw_data_from_web():
    """Chỉ lấy Draw ID và Bộ số từ web, bỏ qua ngày tháng của web"""
    print(f"🌐 Đang kiểm tra dữ liệu mới từ {URL_SOURCE}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(URL_SOURCE, headers=headers, timeout=15)
        
        dfs = pd.read_html(response.content)
        target_df = None
        for d in dfs:
            if d.shape[1] >= 8 and d.shape[0] > 1:
                target_df = d
                break
        
        if target_df is None:
            return []

        raw_results = []
        for _, row in target_df.iterrows():
            try:
                row_str = " ".join([str(x) for x in row.values])
                
                # 1. Lấy Draw ID
                ky_match = re.search(r"(?:#|Kỳ[:\s]*)(\d{4,5})", row_str)
                draw_id = int(ky_match.group(1)) if ky_match else 0
                
                # 2. Lấy Bộ số (chỉ lấy số, bỏ qua ngày)
                nums = [int(s) for s in re.findall(r"\b(\d{1,2})\b", row_str)]
                valid_nums = [n for n in nums if 1 <= n <= 45]
                
                if draw_id > 0 and len(valid_nums) >= 6:
                    # Lấy 6 số cuối cùng (thường là kết quả quay)
                    winning = valid_nums[-6:]
                    raw_results.append({
                        "draw_id": draw_id,
                        "numbers": winning
                    })
            except:
                continue
                
        return raw_results
    except Exception as e:
        print(f"❌ Lỗi kết nối web: {e}")
        return []

# --- MAIN ---
def main(argv=None):
    # 1. Đọc DB hiện tại để lấy mốc (Anchor)
    if not DATA_PATH.exists():
        print("Chưa có file dữ liệu gốc. Vui lòng chạy tool/reset_dates_logic.py để khởi tạo.")
        return

    df = pd.read_csv(DATA_PATH, sep=";")
    
    if df.empty:
        print("File dữ liệu rỗng.")
        return

    # Lấy thông tin kỳ quay cuối cùng trong DB
    # Chắc chắn draw_id là int và date là datetime
    df["draw_id"] = df["draw_id"].astype(int)
    # Parse ngày theo chuẩn YYYY-MM-DD (vì file đã được chuẩn hóa bởi script reset trước đó)
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # Sắp xếp để lấy dòng cuối chuẩn xác
    df = df.sort_values("draw_id")
    last_row = df.iloc[-1]
    
    last_id = int(last_row["draw_id"])
    last_date = last_row["date"]
    
    if pd.isna(last_date):
        print("❌ Lỗi nghiêm trọng: Ngày của kỳ quay cuối cùng trong DB bị lỗi.")
        print("👉 Hãy chạy lại 'python tools/reset_dates_logic.py' để sửa file gốc trước.")
        return

    print(f"📌 DB hiện tại: Kỳ {last_id} - Ngày {last_date.strftime('%Y-%m-%d')}")

    # 2. Lấy dữ liệu thô từ Web
    web_data = fetch_raw_data_from_web()
    
    # Lọc ra các kỳ MỚI HƠN last_id
    new_items = [item for item in web_data if item["draw_id"] > last_id]
    
    if not new_items:
        print("💤 Không có dữ liệu mới.")
        return

    # Sắp xếp tăng dần theo ID để tính toán ngày tuần tự
    new_items.sort(key=lambda x: x["draw_id"])
    
    print(f"✨ Phát hiện {len(new_items)} kỳ quay mới (Từ {new_items[0]['draw_id']} đến {new_items[-1]['draw_id']})")

    # 3. Tính toán ngày & Tạo dòng mới
    rows_to_add = []
    
    # Mốc tính toán hiện tại (bắt đầu từ kỳ cuối trong DB)
    curr_calc_id = last_id
    curr_calc_date = last_date
    
    for item in new_items:
        target_id = item["draw_id"]
        
        # Tính ngày cho target_id dựa trên mốc liền trước
        # (Hàm này xử lý cả việc nhảy cóc nếu web bị thiếu kỳ ở giữa, nhưng vẫn giữ đúng lịch)
        calculated_date = calculate_future_date(curr_calc_id, curr_calc_date, target_id)
        
        # Cập nhật mốc mới
        curr_calc_id = target_id
        curr_calc_date = calculated_date
        
        # Tạo dòng dữ liệu
        new_row = {
            "draw_id": target_id,
            "date": calculated_date,  # Đây là object datetime
            "n_1": item["numbers"][0],
            "n_2": item["numbers"][1],
            "n_3": item["numbers"][2],
            "n_4": item["numbers"][3],
            "n_5": item["numbers"][4],
            "n_6": item["numbers"][5],
        }
        rows_to_add.append(new_row)
        print(f"   + Kỳ {target_id}: Tự động điền ngày {calculated_date.strftime('%Y-%m-%d')}")

    # 4. Ghi vào file
    if rows_to_add:
        df_new = pd.DataFrame(rows_to_add)
        # Gộp vào
        df_final = pd.concat([df, df_new], ignore_index=True)
        
        # Format ngày ra chuỗi chuẩn YYYY-MM-DD để lưu
        df_final["date"] = df_final["date"].dt.strftime("%Y-%m-%d")
        
        df_final.to_csv(DATA_PATH, index=False, sep=";")
        print(f"✅ Đã cập nhật thành công {len(rows_to_add)} kỳ mới vào file.")
    
if __name__ == "__main__":
    main()