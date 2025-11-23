# src/utils/evaluate_models.py

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from core_model.data_prep import load_multi_hot_data
from core_model.features import build_advanced_features_from_multi_hot
from core_model.registry import REGISTRY_PATH


MODELS_DIR = Path("models")
METRICS_PATH = Path("data/model_metrics.parquet")


def load_models_for_version(version: str):
    """
    Load 6 models + scalers cho 1 version nhất định.
    Giống inference.load_models nhưng có tham số version.
    """
    import joblib

    model_dir = MODELS_DIR / version
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    models = {}
    for pos in range(1, 7):
        model_path = model_dir / f"model_pos_{pos}.pkl"
        scaler_path = model_dir / f"scaler_pos_{pos}.pkl"
        models[f"n_{pos}"] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
        }
    return models


def evaluate_version(version: str, X, y_all, n_last: int = 200):
    """
    Đánh giá 1 version trên N bản ghi gần nhất.
    Metric:
        - hit_rate_any : tỉ lệ có ít nhất 1 số trùng
        - hit_rate_exact : tỉ lệ trùng cả 6 số (order-insensitive)
        - mean_hits : trung bình số lượng số trùng
    """
    models = load_models_for_version(version)

    X_eval = X[-n_last:]
    y_eval = y_all[-n_last:]
    n = X_eval.shape[0]

    hits_any = 0
    hits_exact = 0
    total_hits = 0

    for i in range(n):
        x_row = X_eval[i]
        true_nums = set(map(int, y_eval[i]))

        # predict 6 vị trí
        preds = []
        for pos in range(1, 7):
            m = models[f"n_{pos}"]
            x_scaled = m["scaler"].transform([x_row])
            pred = int(m["model"].predict(x_scaled)[0])
            preds.append(pred)

        pred_set = set(preds)
        inter = pred_set & true_nums
        c = len(inter)

        total_hits += c
        if c > 0:
            hits_any += 1
        if c == 6:
            hits_exact += 1

    return {
        "version": version,
        "evaluated_at": datetime.utcnow(),
        "n_samples": n,
        "hit_rate_any": hits_any / n if n else 0.0,
        "hit_rate_exact": hits_exact / n if n else 0.0,
        "mean_hits": total_hits / n if n else 0.0,
    }


def main():
    # 1) Load data & build features
    df, y_all, _ = load_multi_hot_data("data/multi_hot_matrix.csv")
    X, _, _ = build_advanced_features_from_multi_hot(df, y_all)

    # 2) Lấy danh sách version trong thư mục models/
    versions = sorted(
        [p.name for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.startswith("v")]
    )

    metrics = []
    for v in versions:
        print(f"Evaluating version {v} ...")
        m = evaluate_version(v, X, y_all, n_last=200)
        metrics.append(m)

    df_new = pd.DataFrame(metrics)

    # 3) Append vào metrics parquet
    if METRICS_PATH.exists():
        df_old = pd.read_parquet(METRICS_PATH)
        # loại bỏ version trùng, giữ bản mới
        df_old = df_old[~df_old["version"].isin(df_new["version"])]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(METRICS_PATH, index=False)
    print("Saved metrics to", METRICS_PATH)


if __name__ == "__main__":
    main()
