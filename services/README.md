# Services

Core services for the Solana AI Trading System.

## Structure

- **ingest/** - Market data collection (Binance WebSocket — operational)
- **models/** - AI/ML models (classification and NLP — operational; others planned)
- **processing/** - Feature engineering pipeline (planned, not yet implemented)
- **execution/** - Trade execution layer (planned, not yet implemented)

## What Works Today

| Service | Status |
|---------|--------|
| `ingest/connectors/binance_orderbook.py` | Operational |
| `models/classification/` | Operational |
| `models/nlp/` | Operational |
| `models/spoofing_detection/` | Research prototype |
| `models/fusion/` | Partial implementation |
| `processing/` | Not implemented |
| `execution/` | Not implemented |
