import numpy as np
import pandas as pd


# ============================================================
# 1. RECENCY FEATURES (chuẩn xác)
# ============================================================

def compute_recency_features(y_all, max_recency=50):
    """
    Tính recency cho mỗi t (1..T), cho cả 45 số.
    
    Input:
        y_all: numpy array shape (T, 45) — multi-hot matrix
               Mỗi dòng: 45 số, 1 nếu số xuất hiện, 0 nếu không.
    
    Output:
        recency: numpy array shape (T, 45)
                 recency[t][j] = số ngày kể từ lần gần nhất số j+1 xuất hiện
    """

    T, N = y_all.shape  # N = 45
    recency = np.zeros((T, N), dtype=float)

    last_seen = np.full(N, -1, dtype=int)  # -1 = chưa bao giờ xuất hiện

    for t in range(T):
        # update last seen
        for j in range(N):
            if y_all[t, j] == 1:
                last_seen[j] = t

        # compute recency for this time t
        for j in range(N):
            if last_seen[j] == -1:
                recency[t, j] = max_recency
            else:
                recency[t, j] = t - last_seen[j]

    return recency


# ============================================================
# 2. FEATURE FOR NEXT DRAW
# ============================================================

def build_feature_for_next_draw(df, y_all, windows=(30, 60), max_recency=50):
    """
    Xây feature vector theo đúng logic training, cho lần quay T+1.
    """

    T, N = y_all.shape  # N = 45

    # Frequency feature
    freq = []
    for w in windows:
        if T > w:
            freq.append(y_all[T-w:T].mean(axis=0))
        else:
            # Nếu dữ liệu ít hơn cửa sổ, lấy trung bình toàn bộ
            # Nếu T=0 (chưa có dữ liệu), trả về zeros
            if T > 0:
                freq.append(y_all.mean(axis=0))
            else:
                freq.append(np.zeros(N))
                
    freq_feature = np.concatenate(freq)

    # Recency feature
    rec_matrix = compute_recency_features(y_all, max_recency)
    if len(rec_matrix) > 0:
        rec_next = rec_matrix[-1]
    else:
        rec_next = np.full(N, max_recency)

    # Time feature
    if not df.empty:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        # Tránh lỗi nếu last_date là NaT
        if pd.isna(last_date):
            time_features = np.array([T, 0, 0], dtype=float)
        else:
            time_features = np.array([T, last_date.month, last_date.weekday()], dtype=float)
    else:
        time_features = np.array([0, 0, 0], dtype=float)

    # Kết hợp
    x_new = np.concatenate([freq_feature, rec_next, time_features])
    
    # Xử lý NaN cuối cùng (nếu có)
    x_new = np.nan_to_num(x_new, nan=0.0)
    
    return x_new


# ============================================================
# 3. FULL FEATURE MATRIX FOR TRAINING
# ============================================================

def build_advanced_features_from_multi_hot(df, y_all, windows=(30, 60), max_recency=50):
    """
    Build toàn bộ feature cho 100% lịch sử.

    Output:
        X: (T, feature_dim)
        y_all: (T, 45)
        meta_df: pandas DataFrame (gồm cột gốc), align với X
    """

    T, N = y_all.shape
    recency = compute_recency_features(y_all, max_recency)

    X = []

    for t in range(T):
        # frequency window at time t
        freq_t = []
        for w in windows:
            if t <= 0:
                # Tại t=0, chưa có lịch sử, gán 0
                freq_t.append(np.zeros(N))
            elif t - w < 0:
                # Nếu chưa đủ w ngày, lấy trung bình từ 0->t
                slice_data = y_all[:t]
                if slice_data.size == 0:
                    freq_t.append(np.zeros(N))
                else:
                    freq_t.append(slice_data.mean(axis=0))
            else:
                # Đủ w ngày
                freq_t.append(y_all[t-w:t].mean(axis=0))
                
        freq_t = np.concatenate(freq_t)

        rec_t = recency[t]

        # Time feature
        val_date = df.loc[t, "date"]
        if pd.isna(val_date):
            # Giá trị mặc định nếu ngày bị lỗi
            time_t = np.array([t, 0, 0], dtype=float)
        else:
            date_t = pd.to_datetime(val_date)
            time_t = np.array([t, date_t.month, date_t.weekday()], dtype=float)

        # Gộp vector
        row_feat = np.concatenate([freq_t, rec_t, time_t])
        X.append(row_feat)

    X = np.array(X, dtype=float)

    # Bước quan trọng: Thay thế toàn bộ NaN bằng 0.0 để Sklearn không báo lỗi
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # meta_df giữ nguyên các cột gốc (n_1..n_6) để dùng làm nhãn
    meta_df = df.reset_index(drop=True).copy()
    return X, y_all, meta_df