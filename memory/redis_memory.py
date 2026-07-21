import json
import redis
from datetime import datetime

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


def save_turn(turn: dict):
    conversation = {
        # "turn_id": str(redis_client.llen("conversation") + 1,)
        "user": turn['user'],
        "assistant": turn['assistant'],
        "timestamp": datetime.now().isoformat(),
    }

    redis_client.rpush(
        "conversation",
        json.dumps(conversation)
    )


def get_history():
    messages = redis_client.lrange(
        "conversation",
        0,
        -1
    )

    return [
        json.loads(message)
        for message in messages
    ]


def clear_history():
    redis_client.delete("conversation")
