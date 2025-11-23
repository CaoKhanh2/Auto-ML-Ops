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

def build_feature_for_next_draw(df, y_all, windows=(10, 20, 50), max_recency=50):
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
            freq.append(y_all.mean(axis=0))
    freq_feature = np.concatenate(freq)

    # Recency feature
    rec_matrix = compute_recency_features(y_all, max_recency)
    rec_next = rec_matrix[-1]  # dòng cuối cùng

    # Time feature
    last_date = pd.to_datetime(df["date"].iloc[-1])
    time_features = np.array([T, last_date.month, last_date.weekday()], dtype=float)

    # Kết hợp
    x_new = np.concatenate([freq_feature, rec_next, time_features])
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
        y_cols: ["n_1", ..., "n_45"]
    """

    T, N = y_all.shape
    recency = compute_recency_features(y_all, max_recency)

    X = []

    for t in range(T):
        # frequency window at time t
        freq_t = []
        for w in windows:
            if t - w < 0:
                freq_t.append(y_all[:t].mean(axis=0))
            else:
                freq_t.append(y_all[t-w:t].mean(axis=0))
        freq_t = np.concatenate(freq_t)

        rec_t = recency[t]

        date_t = pd.to_datetime(df.loc[t, "date"])
        time_t = np.array([t, date_t.month, date_t.weekday()], dtype=float)

        X.append(np.concatenate([freq_t, rec_t, time_t]))

    X = np.array(X, dtype=float)

    y_cols = [f"n_{i}" for i in range(1, 46)]
    return X, y_all, y_cols
