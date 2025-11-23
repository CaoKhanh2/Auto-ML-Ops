import os
import json
import joblib
from pathlib import Path
from datetime import datetime

from core_model.data_prep import load_multi_hot_data
from core_model.features import build_advanced_features_from_multi_hot


def train_logistic_for_position(X, Y):
    """
    Train model for predicting the number at a single position.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, multi_class="ovr")
    clf.fit(X_scaled, Y)

    return clf, scaler


def train_and_save_model(csv_path, output_dir="models"):
    print("Loading data...")
    df, y_all, y_cols = load_multi_hot_data(csv_path)

    print("Building features...")
    # X = feature matrix (T - max_window rows)
    # Y_multi = multi-hot targets aligned with X
    # meta_df = original df rows aligned with X
    X, Y_multi, meta_df = build_advanced_features_from_multi_hot(df, y_all)

    print("Feature shape:", X.shape)
    print("Meta rows:", meta_df.shape)
    print("Targets:", Y_multi.shape)

    os.makedirs(output_dir, exist_ok=True)

    model_set = {}

    # Train 6 models – one for each column n_1 ... n_6
    for pos in range(1, 7):
        print(f"\nTraining model for position n_{pos} ...")

        # Y cho đúng vị trí, khớp với meta_df (T - max_window)
        Y = meta_df[f"n_{pos}"].astype(int).values

        clf, scaler = train_logistic_for_position(X, Y)

        model_set[f"n_{pos}"] = {
            "model": clf,
            "scaler": scaler
        }

    # Create version
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    print("\nSaving model files...")
    for pos in range(1, 7):
        joblib.dump(model_set[f"n_{pos}"]["model"], model_dir / f"model_pos_{pos}.pkl")
        joblib.dump(model_set[f"n_{pos}"]["scaler"], model_dir / f"scaler_pos_{pos}.pkl")

    # Save registry.json
    registry = {
        "current_version": version,
        "positions": [f"n_{i}" for i in range(1, 7)],
        "path": str(model_dir),
        "created_at": datetime.now().isoformat()
    }

    with open(Path(output_dir) / "registry.json", "w") as f:
        json.dump(registry, f, indent=4)

    print(f"\nDONE: saved models → {model_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core_model.train <csv_path>")
        exit(1)

    csv = sys.argv[1]
    train_and_save_model(csv)
