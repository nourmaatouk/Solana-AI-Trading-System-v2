# Feature Engineering

Feature calculation and technical indicator computation.

## Purpose

Calculates:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (mean, std, skewness)
- Microstructure features (bid-ask spread, order imbalance)
- Sentiment features (from NLP models)
- Network features (from correlation analysis)

## Feature Categories

- **Price features**: Returns, momentum, volatility
- **Volume features**: Volume-weighted metrics
- **Order book features**: Depth, spread, order flow
- **Correlation features**: Inter-token correlations
- **Sentiment features**: Aggregated sentiment scores

## Optimization

- Vectorized calculations using NumPy/Pandas
- Incremental updates for streaming data
- Caching of intermediate results
