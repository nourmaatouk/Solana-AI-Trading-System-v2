# Data Ingestion Service

Real-time market data collection.

## What Is Implemented

### Binance Orderbook Connector

`connectors/binance_orderbook.py` — WebSocket stream for SOL/USDT Level-2 orderbook snapshots.

Collects bids/asks every 100ms and writes aggregated 15-minute features:
- `bid_ask_spread`
- `orderbook_imbalance`
- `top5_bid_volume` / `top5_ask_volume`

Output: `services/models/classification/data/realtime/orderbook_features.csv`

## What Is Not Yet Implemented

The following connectors are planned but do not exist yet:

- OpenBook DEX (on-chain order book)
- Helius (enriched blockchain data)
- Direct Solana blockchain events

Reddit sentiment collection is handled by the NLP service (`services/models/nlp/`), not this ingest layer.
