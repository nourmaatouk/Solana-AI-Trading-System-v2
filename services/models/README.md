# Models

AI/ML model implementations for the Solana trading pipeline.

## Status

| Module | Status | Notes |
|--------|--------|-------|
| `classification/` | Operational | XGBoost H4/H8/H16, 64% accuracy |
| `nlp/` | Operational | FinBERT sentiment, GDELT + RSS + Reddit |
| `spoofing_detection/` | Research prototype | Synthetic labels, not production-ready |
| `fusion/` | Partial | Aggregator and contracts exist; full integration pending |
| `spatial/` | Planned | GNN architecture — not yet implemented |
| `temporal/` | Planned | LSTM/Transformer — not yet implemented |
| `rl/` | Planned | PPO/DQN execution agent — not yet implemented |

## Roadmap

The following phases are planned but not yet built:

**Phase 1 — Social Sentiment (Reddit)**
Reddit scraper is implemented in `nlp/` (`RedditFetcher`). Subreddits: `r/solana`, `r/CryptoCurrency`, `r/CryptoMarkets`.

**Phase 2 — Hawkes Process (Volatility Intensity)**
MLE script for market self-excitation modelling. Not yet implemented.

**Phase 3 — TEN-GNN Spatial Fusion**
TEN and GNN prototypes exist in `spoofing_detection/`. Integration into the classification pipeline is not yet done.

**Phase 4 — RL Execution Agent (PPO)**
Reinforcement learning execution layer. Not yet implemented.

## Integration Checklist

- [x] Triple-Barrier Labeling (+2.0% / -1.5%)
- [x] XGBoost H4/H8/H16 classifiers
- [x] FinBERT NLP sentiment pipeline
- [x] Reddit fetcher
- [ ] Hawkes Intensity MLE
- [ ] GNN feature extraction layer
- [ ] RL execution orchestrator
