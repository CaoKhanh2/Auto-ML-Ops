"""Batch script to predict the next draw using the latest trained models."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure `src/` is on sys.path when running directly
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = SRC_DIR.parent

from core_model.data_prep import load_multi_hot_data
from core_model.inference import predict_next_draw
from storage.redis_client import save_next_draw, save_prediction # <--- Import thêm
from core_model.registry import get_current_version_meta # <--- Import thêm

def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict next draw from historical data")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data/multi_hot_matrix.csv"),
        help="Path to multi_hot_matrix.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data/output/next_prediction.json"),
        help="Where to save the prediction JSON",
    )

    args = parser.parse_args(argv)

    df, y_all, _ = load_multi_hot_data(args.data)
    preds = predict_next_draw(df, y_all)

    # 1. Lưu JSON file (như cũ)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"prediction": preds}, indent=2))

    # 2. Lưu vào Redis để Dashboard hiển thị (MỚI)
    meta = get_current_version_meta()
    version = meta.get("current_version", "unknown")
    
    redis_payload = {
        "numbers": preds,
        "version": version,
        "ts": datetime.utcnow().isoformat(),
        "source": "batch_airflow"
    }
    save_next_draw(redis_payload)
    print(f"Saved to Redis: {redis_payload}")

    print(f"Next draw prediction: {preds}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()