import os
import faust
import numpy as np
import pandas as pd

from core_model.inference import predict_from_features
from storage.redis_client import save_prediction, save_next_draw
from storage.parquet_sink import append_prediction
from storage.log_parquet import append_log

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka://kafka:9092")
IN_TOPIC_NAME = os.getenv("KAFKA_FEATURE_TOPIC", "lottery_features")
OUT_TOPIC_NAME = os.getenv("KAFKA_PRED_TOPIC", "lottery_predictions")

app = faust.App(
    "predict-number-streaming",
    broker=KAFKA_BROKER,
    value_serializer="json",
)

in_topic = app.topic(IN_TOPIC_NAME, value_type=dict)
out_topic = app.topic(OUT_TOPIC_NAME, value_type=dict)


@app.agent(in_topic)
async def processor(stream):
    async for event in stream:
        try:
            draw_id = int(event["draw_id"])
            features = event["features"]
            ts = event.get("timestamp")

            probs = predict_from_features(features)
            idx = np.argsort(-probs)[:6]

            payload = {
                "draw_id": draw_id,
                "timestamp": ts,
                "top_numbers": (idx + 1).tolist(),
                "top_probs": probs[idx].tolist(),
            }

            await out_topic.send(value=payload)
            save_prediction(draw_id, payload)
            save_next_draw(payload)

            df = pd.DataFrame([payload])
            append_prediction(df)
        except Exception as exc:
            append_log(
                "stream_error",
                {"stage": "process_event", "error": str(exc), "event": event},
            )


if __name__ == "__main__":
    app.main()
