import json
import time
import random
from confluent_kafka import Producer

KAFKA_BROKER = "kafka:9092"
TOPIC = "raw_features"

producer = Producer({
    "bootstrap.servers": KAFKA_BROKER
})

FEATURE_DIM = 27   # Điều này phù hợp với model của bạn


def generate_feature_vector():
    return [round(random.random(), 6) for _ in range(FEATURE_DIM)]


if __name__ == "__main__":
    print("Data generator started.")
    while True:
        features = generate_feature_vector()
        payload = {"features": features}

        producer.produce(
            TOPIC,
            json.dumps(payload).encode("utf-8")
        )
        producer.flush()

        print("Sent:", payload)
        time.sleep(1)
