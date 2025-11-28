import pandas as pd
from pathlib import Path
from datetime import timedelta

# Đường dẫn file dữ liệu
DATA_PATH = Path("data/multi_hot_matrix.csv")

def get_next_draw_date(current_date):
    """Tính ngày quay tiếp theo (Thứ 4, 6, CN)"""
    # 0: Mon, 2: Wed, 4: Fri, 6: Sun
    weekday = current_date.weekday()
    
    if weekday == 2:   # Thứ 4 -> Thứ 6 (+2 ngày)
        return current_date + timedelta(days=2)
    elif weekday == 4: # Thứ 6 -> Chủ Nhật (+2 ngày)
        return current_date + timedelta(days=2)
    elif weekday == 6: # Chủ Nhật -> Thứ 4 tuần sau (+3 ngày)
        return current_date + timedelta(days=3)
    else:
        # Trường hợp ngày gốc bị sai lệch, mặc định cộng 2 để tiếp tục
        return current_date + timedelta(days=2)

def main():
    print(f"Đang đọc file: {DATA_PATH}")
    # Đọc file, xử lý sơ bộ
    df = pd.read_csv(DATA_PATH, sep=";")
    
    # Chuyển đổi cột date, những dòng lỗi hoặc trống sẽ thành NaT
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    
    # Sắp xếp theo draw_id để đảm bảo thứ tự
    df = df.sort_values("draw_id").reset_index(drop=True)
    
    print("Đang tính toán và điền ngày thiếu...")
    count_filled = 0
    
    # Duyệt qua từng dòng để điền ngày thiếu
    for i in range(len(df)):
        if pd.isna(df.loc[i, "date"]):
            # Nếu là dòng đầu tiên mà thiếu thì không tính được (cần dữ liệu mồi)
            if i == 0:
                continue
                
            # Lấy ngày của kỳ trước đó
            prev_date = df.loc[i-1, "date"]
            
            if pd.notna(prev_date):
                # Tính ngày mới dựa trên quy luật 4-6-CN
                new_date = get_next_draw_date(prev_date)
                df.loc[i, "date"] = new_date
                count_filled += 1
    
    # Format lại thành chuỗi DD/MM/YYYY để lưu vào CSV
    df["date"] = df["date"].dt.strftime("%d/%m/%Y")
    
    # Lưu đè lại file cũ
    df.to_csv(DATA_PATH, index=False, sep=";")
    print(f"✅ Đã điền tự động {count_filled} ngày bị thiếu.")
    print(f"Lưu thành công tại: {DATA_PATH}")

if __name__ == "__main__":
    main()