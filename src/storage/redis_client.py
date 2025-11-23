import os
import json
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

def get_redis():
    """
    Trả về Redis client.
    decode_responses=True => dữ liệu trả về là string.
    """
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )

# -------------------------------
# WRAPPER HỖ TRỢ LƯU PREDICTION
# -------------------------------

def save_prediction(draw_id, payload):
    r = get_redis()
    r.set(f"prediction:{draw_id}", json.dumps(payload))


def get_prediction(draw_id):
    r = get_redis()
    data = r.get(f"prediction:{draw_id}")
    return json.loads(data) if data else None


def save_next_draw(payload):
    r = get_redis()
    r.set("next_draw", json.dumps(payload))


def get_next_draw():
    r = get_redis()
    data = r.get("next_draw")
    return json.loads(data) if data else None
