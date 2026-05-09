# Data Connectors

## Implemented

### `binance_orderbook.py`

WebSocket connector for the Binance SOL/USDT Level-2 orderbook.

- Streams live bids/asks at 100ms intervals
- Aggregates into 15-minute orderbook features
- Handles reconnection on disconnect
- Writes to `services/models/classification/data/realtime/latest_orderbook_features.csv`

## Not Yet Implemented

The following connectors are planned but have no code:

- `openbook_connector.py` — Solana OpenBook DEX on-chain order book
- `helius_connector.py` — Helius RPC enriched blockchain events
- `solana_connector.py` — Direct Solana blockchain data
