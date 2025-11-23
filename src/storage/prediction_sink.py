# src/storage/prediction_sink.py

import json
from datetime import datetime
from pathlib import Path

from kafka import KafkaConsumer
import pandas as pd

from storage.redis_client import save_prediction, save_next_draw
from core_model.registry import get_current_version_meta

KAFKA_BROKER = "kafka:9092"
TOPIC = "predictions"

HISTORY_PATH = Path("data/predictions_history.parquet")


def append_history(record: dict):
    """
    Ghi 1 bản ghi dự đoán vào lịch sử (parquet).
    """
    df_new = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        df_old = pd.read_parquet(HISTORY_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(HISTORY_PATH, index=False)


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="prediction-sink-group",
    )

    print("Prediction sink started. Waiting for messages...")

    while True:
        for msg in consumer:
            payload = msg.value
            # giả định payload có draw_id, numbers, source,...
            draw_id = payload.get("draw_id")
            numbers = payload.get("numbers") or payload.get("prediction") or []
            source = payload.get("source", "stream")

            # lấy version hiện tại từ registry
            meta = get_current_version_meta()
            version = meta.get("current_version")

            # Lưu vào Redis (cho API / dashboard đọc)
            if draw_id is not None:
                save_prediction(draw_id, {
                    "draw_id": draw_id,
                    "numbers": numbers,
                    "version": version,
                    "source": source,
                    "ts": datetime.utcnow().isoformat(),
                })

            # nếu là dự đoán cho kỳ tiếp theo thì ghi cả key "next_draw"
            if payload.get("is_next", False):
                save_next_draw({
                    "numbers": numbers,
                    "version": version,
                    "ts": datetime.utcnow().isoformat(),
                })

            # Lưu vào lịch sử parquet
            history_rec = {
                "event_time": datetime.utcnow(),
                "draw_id": draw_id,
                "numbers": numbers,
                "source": source,
                "model_version": version,
            }
            append_history(history_rec)

            print("Saved prediction:", history_rec)


if __name__ == "__main__":
    main()
