import json
import asyncio
from confluent_kafka import Consumer, Producer
from core_model.inference import predict_from_features


KAFKA_BROKER = "kafka:9092"
INPUT_TOPIC = "raw_features"
OUTPUT_TOPIC = "predictions"


def create_consumer():
    return Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "predict-worker",
        "auto.offset.reset": "earliest"
    })


def create_producer():
    return Producer({
        "bootstrap.servers": KAFKA_BROKER
    })


async def handle_message(msg, producer):
    data = json.loads(msg.value().decode("utf-8"))
    
    features = data["features"]  # list of numbers
    preds = predict_from_features(features)

    out = {
        "prediction": preds,
        "source": data
    }

    producer.produce(
        OUTPUT_TOPIC,
        json.dumps(out).encode("utf-8")
    )
    producer.flush()


async def main_loop():
    consumer = create_consumer()
    producer = create_producer()
    consumer.subscribe([INPUT_TOPIC])

    print("Streaming worker started...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            if msg.error():
                print("Kafka Error:", msg.error())
                continue
            
            await handle_message(msg, producer)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    asyncio.run(main_loop())
