from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json

from core_model.inference import predict_from_features, predict_next_draw
from core_model.data_prep import load_multi_hot_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "multi_hot_matrix.csv"

app = FastAPI(title="Predict Number API")
rds = redis.Redis(host="redis", port=6379, db=0)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_from_features(req.features)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_next_draw")
def api_predict_next_draw():
    if not DEFAULT_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Missing data file: {DEFAULT_DATA_PATH}")

    df, y_all, _ = load_multi_hot_data(DEFAULT_DATA_PATH)
    result = predict_next_draw(df, y_all)
    return {"prediction": result}


@app.get("/prediction/{item_id}")
def get_prediction(item_id: str):
    key = f"prediction:{item_id}"
    val = rds.get(key)
    if not val:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return json.loads(val)