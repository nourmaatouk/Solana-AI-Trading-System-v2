# Data Ingestion Service

Real-time market data collection from multiple sources.

## Structure

- **connectors/** - Data source connectors

## Purpose

Collects data from:
- Binance (spot market data)
- OpenBook DEX (on-chain order book)
- Helius (enriched blockchain data)
- Reddit (sentiment data)
- Solana blockchain (on-chain events)

## Processing Pipeline

1. Data collection from source
2. Validation and normalization
3. Storage in TimescaleDB/Redis
4. Event publishing to processing pipeline

## Real-time Updates

- WebSocket connections for live data
- Order book snapshots and updates
- Trade execution events
- Sentiment score updates
