import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import FusionAggregator from the same package
try:
    from .fusion_aggregator import FusionAggregator
except Exception:
    # Fallback if module is executed from different cwd
    from fusion_aggregator import FusionAggregator


class StreamingFusionAdapter:
    """A simple, pluggable streaming adapter for fusion.

    - Uses in-memory asyncio queues by default for classification, sentiment, spoofing.
    - Buffers signals for a short window then calls `FusionAggregator.fuse_batch`.
    - Publishes fused decisions to an output queue.

    This is intentionally lightweight for local testing; replace queues with
    Kafka/Redis consumers/producers for production.
    """

    def __init__(self,
                 classification_q: Optional[asyncio.Queue] = None,
                 sentiment_q: Optional[asyncio.Queue] = None,
                 spoofing_q: Optional[asyncio.Queue] = None,
                 out_q: Optional[asyncio.Queue] = None,
                 buffer_seconds: float = 2.0,
                 aggregator: Optional[FusionAggregator] = None):
        self.class_q = classification_q or asyncio.Queue()
        self.sent_q = sentiment_q or asyncio.Queue()
        self.spoof_q = spoofing_q or asyncio.Queue()
        self.out_q = out_q or asyncio.Queue()
        self.buffer_seconds = buffer_seconds
        self.aggregator = aggregator or FusionAggregator()
        self._running = False

    async def _drain_queue(self, q: asyncio.Queue):
        """Drain all available items from queue without waiting."""
        items = []
        while True:
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                items.append(item)
                q.task_done()
        return items

    async def _collect_buffers(self):
        # Collect available items from each queue
        class_items = await self._drain_queue(self.class_q)
        sent_items = await self._drain_queue(self.sent_q)
        spoof_items = await self._drain_queue(self.spoof_q)
        return class_items, sent_items, spoof_items

    async def stream_loop(self):
        """Main loop: buffer signals, fuse, publish."""
        logger.info("StreamingFusionAdapter starting stream loop")
        self._running = True
        try:
            while self._running:
                # Wait buffer_seconds to gather signals
                await asyncio.sleep(self.buffer_seconds)

                class_items, sent_items, spoof_items = await self._collect_buffers()

                if not (class_items or sent_items or spoof_items):
                    # Nothing to do this cycle
                    continue

                # Convert list of dicts to simple DataFrame-like lists for aggregator
                # The aggregator expects DataFrames; for simplicity we pass lists
                # and the aggregator supports list/dict inputs in local test harness.
                try:
                    fused_df = self.aggregator.fuse_batch(
                        classification_df=class_items if class_items else None,
                        sentiment_df=sent_items if sent_items else None,
                        spoofing_df=spoof_items if spoof_items else None,
                        join_key="timestamp",
                    )
                except Exception as e:
                    logger.exception("Fusion aggregator failed: %s", e)
                    continue

                # fused_df expected to be a pandas DataFrame or list of dicts
                # Normalize to list of dicts
                try:
                    rows = fused_df.to_dict(orient="records")
                except Exception:
                    # assume it's already list-like
                    rows = list(fused_df)

                # Publish each fused decision
                for row in rows:
                    await self.out_q.put(row)
                    logger.info("Published fused decision: %s", row)

        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        finally:
            self._running = False
            logger.info("StreamingFusionAdapter stopped")

    async def start(self):
        self._task = asyncio.create_task(self.stream_loop())
        return self._task

    async def stop(self):
        self._running = False
        if hasattr(self, "_task"):
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
