# Raw Data

Raw market data and source data before processing.

## Purpose

Stores unmodified data from:
- Binance API (OHLCV candles, order book)
- Solana blockchain (on-chain events, transactions)
- OpenBook DEX (order book, fills)
- Helius (enriched blockchain data)
- Reddit (sentiment data)

## Format

Data is stored as:
- JSON files
- CSV files
- Parquet for bulk exports
- Database exports

## Data Pipeline

Raw data → Validation → Processing → Feature Engineering → Training
