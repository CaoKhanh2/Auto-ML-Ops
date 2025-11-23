import json
import joblib
from pathlib import Path
import numpy as np

from core_model.features import build_feature_for_next_draw


REGISTRY_PATH = Path("models/registry.json")


# ---------------------------
# LOAD MODELS (multi-model)
# ---------------------------

def load_models():
    """
    Load all 6 models from the version specified in registry.json.
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("registry.json not found. Please run training first.")

    reg = json.loads(REGISTRY_PATH.read_text())

    version = reg.get("current_version")
    model_dir = Path(reg.get("path", f"models/{version}"))

    models = {}

    for pos in range(1, 7):
        model_path = model_dir / f"model_pos_{pos}.pkl"
        scaler_path = model_dir / f"scaler_pos_{pos}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing: {model_path}")

        models[f"n_{pos}"] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path)
        }

    return models


MODELS = load_models()


# ---------------------------
# INFERENCE
# ---------------------------

def predict_from_features(x_new):
    """
    Predict 6 positions from input feature vector.
    """
    x_new = np.array(x_new)

    preds = []
    for pos in range(1, 7):
        m = MODELS[f"n_{pos}"]
        scaler = m["scaler"]
        model = m["model"]

        x_scaled = scaler.transform([x_new])
        pred = model.predict(x_scaled)[0]
        preds.append(int(pred))

    return preds


def predict_next_draw(df, y_all):
    """
    Build next feature vector and predict.
    """
    x_new = build_feature_for_next_draw(df, y_all)
    return predict_from_features(x_new)
