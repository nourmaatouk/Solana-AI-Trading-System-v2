# Fusion Models

**Status: Partial implementation.**

Infrastructure exists (`src/fusion_aggregator.py`, `src/fusion_contracts.py`, `src/stream_adapter.py`) but full integration with the classification and NLP pipelines is not yet complete.

## What Exists

- `src/fusion_aggregator.py` — Weighted signal aggregator
- `src/fusion_contracts.py` — Shared schema contracts for fused signals
- `src/stream_adapter.py` — Streaming adapter for real-time fusion
- `tests/test_fusion_integration.py` — Integration tests
- `run_streaming_demo.py` — Demo runner

## What Is Missing

- Live integration with `models/classification/` output
- Live integration with `models/nlp/` sentiment output
- RL policy hookup (Phase 4)
- Attention-based fusion (planned)
