import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FusionConsumer:
    """Consume fused decisions and forward to orchestrator or log them.

    Usage:
      consumer = FusionConsumer(fusion_queue)
      asyncio.run(consumer.consume_forever())

    The consumer expects decision objects as dicts or dataclasses with at
    least these keys: `timestamp`, `action`, `fused_score`, `risk_level`,
    `alignment_score`, and optional `reasoning`.
    """

    def __init__(self, fusion_queue: Optional[asyncio.Queue] = None, orchestrator=None):
        # If no queue provided, create an in-memory queue for testing
        self.fusion_queue = fusion_queue or asyncio.Queue()
        self.orchestrator = orchestrator
        self._history = []

    async def process_decision(self, decision):
        """Process a single decision object."""
        # Normalize to dict if it's a dataclass-like object
        try:
            d = dict(decision)
        except Exception:
            d = decision

        # store history (lightweight)
        self._history.append({
            "timestamp": d.get("timestamp"),
            "action": d.get("action"),
            "score": d.get("fused_score"),
            "risk": d.get("risk_level"),
            "alignment": d.get("alignment_score"),
        })

        # Forward to orchestrator if available
        if self.orchestrator and hasattr(self.orchestrator, "process_decision"):
            try:
                result = await self.orchestrator.process_decision(
                    action=d.get("action"),
                    confidence=abs(d.get("fused_score", 0)),
                    risk_level=d.get("risk_level"),
                    alignment_score=d.get("alignment_score"),
                    reasoning=d.get("reasoning", {}),
                )
                logger.info("Orchestrator accepted decision: %s", result)
                return result
            except Exception as e:
                logger.exception("Orchestrator processing failed: %s", e)
                return {"status": "orchestrator_error", "error": str(e)}
        else:
            # No orchestrator available: log and return
            logger.info("Decision received (no orchestrator): %s", d)
            return {"status": "logged"}

    async def consume_forever(self):
        """Continuously consume decisions from the queue."""
        logger.info("FusionConsumer started, waiting for decisions...")
        while True:
            decision = await self.fusion_queue.get()
            try:
                await self.process_decision(decision)
            finally:
                self.fusion_queue.task_done()

    def get_history(self, limit: int = 100):
        return self._history[-limit:]


# If run directly, perform a small self-test
if __name__ == "__main__":
    async def _self_test():
        q = asyncio.Queue()
        consumer = FusionConsumer(q)

        # push a sample decision
        sample = {
            "timestamp": "2026-05-04T10:00:00",
            "action": "BUY",
            "fused_score": 0.42,
            "risk_level": "LOW",
            "alignment_score": 0.6,
            "reasoning": {"notes": "synthetic test"},
        }
        await q.put(sample)

        # run consumer for one item
        task = asyncio.create_task(consumer.consume_forever())

        # wait until queue processed
        await q.join()
        task.cancel()

        print("Self-test history:", consumer.get_history())

    asyncio.run(_self_test())
