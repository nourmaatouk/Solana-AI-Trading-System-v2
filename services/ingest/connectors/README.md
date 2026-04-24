# Data Connectors

Data source connector implementations.

## Purpose

Each connector implements:
- Connection establishment and health monitoring
- Data fetching and transformation
- Error handling and retries
- Metadata enrichment

## Connectors

- **BinanceConnector** - Binance API integration
- **OpenBookConnector** - Solana OpenBook DEX
- **HeliusConnector** - Helius RPCs for enriched data
- **RedditConnector** - Sentiment data extraction
- **SolanaConnector** - Direct blockchain data

## Features

- Automatic reconnection on failure
- Rate limit handling
- Data deduplication
- Timestamp synchronization
