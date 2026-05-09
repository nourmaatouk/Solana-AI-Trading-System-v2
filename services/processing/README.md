# Data Processing Service

**Status: Not yet implemented.**

Feature engineering and pipeline orchestration are planned for a future phase. Currently, feature calculation is handled inside each model service directly:

- Orderbook features: `services/ingest/connectors/binance_orderbook.py`
- Sentiment features: `services/models/nlp/src/aggregation/feature_aggregator.py`
- Realtime feature merge: `services/models/classification/realtime_merge_features.py`

When this service is built it will centralize those concerns into a shared processing layer.
