"""Run a small streaming demo: mock producers -> StreamingFusionAdapter -> FusionConsumer

This is an in-process streaming demo that doesn't require external Kafka/Redis.
Run it to see fused decisions flow through the system.
"""
import asyncio
import random
from datetime import datetime, timedelta
from pprint import pprint

# Ensure imports from repository
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.models.fusion.src.stream_adapter import StreamingFusionAdapter
from apps.agent.fusion_consumer import FusionConsumer


async def mock_classification_producer(q: asyncio.Queue, n: int = 8, start_ts=None):
    start_ts = start_ts or datetime.utcnow()
    for i in range(n):
        ts = (start_ts + timedelta(hours=4 * i)).isoformat()
        item = {
            "timestamp": ts,
            "prediction": random.random(),
            "signal": random.choice(["BUY", "HOLD", "SELL"]),
            "confidence": random.uniform(0.6, 0.95),
            "classification_confidence": random.uniform(0.6, 0.95),
        }
        await q.put(item)
        await asyncio.sleep(0.1)


async def mock_sentiment_producer(q: asyncio.Queue, n: int = 33, start_ts=None):
    start_ts = start_ts or datetime.utcnow()
    for i in range(n):
        ts = (start_ts + timedelta(minutes=15 * i)).isoformat()
        item = {
            "timestamp": ts,
            "sentiment_score": random.uniform(-1, 1),
            "sentiment_label": random.choice(["POSITIVE", "NEUTRAL", "NEGATIVE"]),
            "relevance_score": random.uniform(0.5, 1.0),
            "sentiment_relevance": random.uniform(0.0, 1.0),
        }
        await q.put(item)
        await asyncio.sleep(0.02)


async def mock_spoofing_producer(q: asyncio.Queue, n: int = 50, start_ts=None):
    start_ts = start_ts or datetime.utcnow()
    for i in range(n):
        ts = (start_ts + timedelta(seconds=60 * i)).isoformat()
        item = {
            "timestamp": ts,
            "confidence": random.choice([0.0, 0.0, random.uniform(0, 1)]),
            "severity": random.choice(["low", "medium", "high"]),
            "spoofing_confidence": random.choice([0.0, random.uniform(0, 1)]),
        }
        await q.put(item)
        await asyncio.sleep(0.01)


async def main_demo():
    class_q = asyncio.Queue()
    sent_q = asyncio.Queue()
    spoof_q = asyncio.Queue()
    out_q = asyncio.Queue()

    adapter = StreamingFusionAdapter(
        classification_q=class_q,
        sentiment_q=sent_q,
        spoofing_q=spoof_q,
        out_q=out_q,
        buffer_seconds=1.0,
    )

    consumer = FusionConsumer(out_q)

    # Start adapter and consumer
    adapter_task = await adapter.start()
    consumer_task = asyncio.create_task(consumer.consume_forever())

    # Start mock producers
    producers = [
        asyncio.create_task(mock_classification_producer(class_q)),
        asyncio.create_task(mock_sentiment_producer(sent_q)),
        asyncio.create_task(mock_spoofing_producer(spoof_q)),
    ]

    # Collect some fused outputs for a short duration
    try:
        # Wait until producers complete
        await asyncio.gather(*producers)

        # allow adapter to process remaining buffer
        await asyncio.sleep(3)

        # drain out_q
        results = []
        while not out_q.empty():
            results.append(await out_q.get())

        print("\n=== Demo fused decisions (sample) ===")
        pprint(results[:10])

    finally:
        await adapter.stop()
        consumer_task.cancel()


if __name__ == "__main__":
    asyncio.run(main_demo())
