# Solana AI Trading System — Models

End-to-end ML pipeline that ingests market data, news, and Binance L2 orderbook
flow, produces a directional signal for SOL/USDT, fuses model/sentiment/spoofing
context into a final decision, and simulates that decision through paper trading.

It does **not** execute real Solana trades yet. Wallet integration, RL/agent
orchestration, and Solana on-chain execution are still future work.

### Current validated state (2026-05-13)

- Fresh SOL/BTC Binance REST candles work.
- Fresh Binance partial-depth orderbook collection works through `realtime_orderbook_collector.py`.
- `run_realtime_pipeline.py --skip-sentiment --fusion --fusion-aggregate --paper-trade --timeout 60` completed with `Pipeline status: ok`.
- `realtime_predictor.py` produced a real XGBoost signal: `DOWN`, `P(up)=0.2566`.
- `realtime_fusion_predictor.py` ran the XGB+DNN stacker: `UP`, `P(up)=0.7158`.
- `realtime_spoofing_detector.py` used fresh orderbook rows: `status=ok`, TEN spoof probability `0.0713`.
- `realtime_fusion_aggregator.py` wrote `last_fused_decision.json`: `HOLD`, score `-0.1293`.
- `realtime_paper_trader.py` consumed `last_fused_decision.json`.

Recent local checks:

```bash
python -m pytest services/models/classification/tests -q
# 69 passed

python -m pytest services/models/fusion/tests -q
# 22 passed, 1 skipped
```

---

## Architecture

```
Layer 1 — Data Sources
  ├── Binance REST API      (SOL/USDT + BTC/USDT 15-min OHLCV)
  ├── Binance WebSocket     (SOL/USDT partial-depth L2 orderbook)
  ├── GDELT API             (global event/news stream)
  ├── RSS Feeds             (CoinDesk, CoinTelegraph, Decrypt, Solana.com …)
  ├── Reddit JSON API       (r/solana, r/CryptoCurrency, r/CryptoMarkets)
  ├── OpenBook DEX WebSocket (L2/L3 order book — planned, not connected)
  └── Helius RPC WebSocket  (on-chain events — planned)

Layer 2 — Data Processing
  ├── Technical features    (RSI, MACD, Bollinger, ATR, z-scores, BTC co-features)
  ├── Sentiment pipeline    (FinBERT scoring → 15-min bucket aggregation)
  ├── Order-book features   (imbalance, spread, bid/ask volume — operational)
  ├── Triple-barrier labels (+2 % TP / −1.5 % SL over H4/H8/H16 windows)
  └── Hawkes Process MLE    (realtime spoofing gate; not yet a training feature)

Layer 3 — Models
  ├── XGBoost classifier    (H4/H8/H16 binary UP/DOWN, ~64 % accuracy)
  ├── FinBERT sentiment     (ProsusAI/finbert, 3-class → score [-1, 1])
  ├── TEN-GNN spoofing      (Temporal Encoder + GNN, synthetic labels)
  ├── Fusion aggregator     (wired into realtime pipeline via --fusion-aggregate)
  ├── DNN fusion predictor  (XGB+DNN stacker; TEN only if metadata supports it)
  ├── Temporal encoder (TEN)(LSTM/Transformer stub — planned)
  └── Spatial encoder (GNN) (graph anomaly stub — planned)

Layer 4 — Agent Layer
  ├── Paper trader          (simulator — implemented)
  ├── RL execution agent    (PPO — trained, currently disabled in fusion)
  ├── ElizaOS orchestrator  (implemented as post-decision narrator; never on critical path)
  ├── Ollama Llama 3        (planned)
  └── ChromaDB memory       (planned)

Layer 5 — Execution
  └── Solana execution layer (TypeScript Guardian + AutoExecutor + Phantom signing — implemented;
                              on-chain Anchor Guardian — planned)
```

---

## Directory Structure

```
services/models/
├── classification/          # XGBoost pipeline (Layer 2 + 3)
│   ├── 01_collect_sol_data_15m.py     # Binance OHLCV download
│   ├── 02_build_technical_features.py # Technical indicators
│   ├── 02b_add_btc_features.py        # BTC co-features
│   ├── 06_merge_sentiment_features.py # Attach sentiment to OHLCV
│   ├── 06b_merge_orderbook_features.py# Attach order-book features
│   ├── 07_prepare_no_leak_data.py     # Split + scaler + label
│   ├── 08_train_xgboost_no_leak.py    # Train H4/H8/H16 models
│   ├── 20_walk_forward_validation.py  # Walk-forward OOS test
│   ├── 21_xgboost_curves.py           # Learning curves
│   ├── 22_realtime_classifier_dashboard.py
│   ├── 23_backtest.py                 # SL/TP position backtest
│   ├── 24_gate_sweep.py               # Confidence gate sweep
│   ├── 25_fusion_backtest.py          # XGB + DNN + TEN fusion backtest
│   ├── config.py                      # Single source of truth for constants
│   ├── realtime_btc_fetcher.py        # BTC OHLCV realtime fetch
│   ├── realtime_feature_builder.py    # Technical features (realtime)
│   ├── realtime_fusion_predictor.py   # Multi-model fusion inference
│   ├── realtime_market_fetcher.py     # SOL OHLCV realtime fetch
│   ├── realtime_merge_features.py     # Merge tech + sentiment + OB
│   ├── realtime_orderbook_collector.py# WebSocket OB collector (separate process)
│   ├── realtime_paper_trader.py       # Paper trading simulator
│   ├── realtime_predictor.py          # XGBoost inference + signal JSON
│   ├── realtime_sentiment_fetcher.py  # GDELT + RSS → sentiment
│   ├── realtime_spoofing_detector.py  # TEN-GNN gate (realtime)
│   ├── run_realtime_pipeline.py       # Orchestrator (15-min cron)
│   ├── train_dnn_fusion.py            # DNN fusion layer training
│   ├── train_ten_fusion.py            # TEN fusion layer training
│   ├── data/processed/                # NPY splits, scalers, feature_columns.json
│   ├── data/raw/                      # Raw OHLCV CSVs
│   ├── data/realtime/                 # Live pipeline state (JSON/CSV)
│   └── tests/                        # Pytest test suite (69 passing locally)
├── nlp/                     # Sentiment pipeline (Layer 1 + 2 + 3)
│   ├── src/
│   │   ├── config.py                  # Pydantic settings (env-overridable)
│   │   ├── fetchers/                  # GDELT, RSS, Reddit fetchers
│   │   ├── processing/                # FinBERT, text cleaner, sentiment pipeline
│   │   ├── aggregation/               # 15-min bucket aggregator
│   │   ├── contracts/                 # Layer 2 DataFrame contracts
│   │   ├── models/                    # FinBERT wrapper + XGBoost trainer
│   │   └── storage/                   # CSV, Redis, TimescaleDB writers
│   └── tests/                        # 33 pytest tests
├── fusion/                  # Signal fusion contracts + aggregator (Layer 3)
│   ├── src/
│   │   ├── fusion_aggregator.py       # Weighted signal fusion
│   │   ├── fusion_contracts.py        # Typed signal dataclasses
│   │   └── stream_adapter.py          # Streaming integration adapter
│   └── tests/
├── spoofing_detection/      # TEN-GNN spoofing (Layer 3 — research prototype)
│   ├── src/
│   │   ├── models/                    # TEN, GNN, TEN-GNN architectures
│   │   ├── data/                      # LOBSTER/Kaggle loaders, labeling
│   │   ├── training/                  # Trainer
│   │   ├── evaluation/                # Metrics, ablation, OOD, flash crash sim
│   │   ├── deployment/                # Inference engine, Hawkes updater, quantize
│   │   └── explainability/            # SHAP, integrated gradients, reports
│   └── tests/
├── models/                  # Saved model artifacts
│   ├── xgboost_h4.json
│   ├── xgboost_h8.json
│   ├── xgboost_h16.json
│   ├── walk_forward_results.json
│   ├── backtest_results.json
│   └── gate_sweep_results.json
├── integration/             # End-to-end integration tests
├── scripts/                 # Utility / validation scripts
├── tests/                   # Cross-cutting audit tests (security, ML quality)
├── rl/                      # RL execution agent — PPO trained (2M+ steps), currently
│                            # disabled in fusion (WEIGHT_RL=0.0) due to OOS regime mismatch
├── spatial/                 # GNN spatial encoder (stub — not yet implemented)
└── temporal/                # TEN temporal encoder (stub — not yet implemented)
```

---

## Module Status

| Module | Layer | Status | Notes |
|--------|-------|--------|-------|
| `classification/` | 2, 3 | **Operational** | Fresh OHLCV/orderbook path, XGBoost, dashboard, paper trader |
| `nlp/` | 1, 2, 3 | **Operational but slow live** | FinBERT + GDELT + RSS + Reddit; GDELT/HF can timeout |
| `fusion/` | 3 | **Operational** | Aggregator wired via `--fusion-aggregate`; DNN fusion predictor runs |
| `spoofing_detection/` | 3 | Research prototype | Realtime Hawkes/TEN gate works; labels are synthetic/research-grade |
| `spatial/` | 3 | Stub | GNN architecture — not yet implemented |
| `temporal/` | 3 | Stub | LSTM/Transformer — not yet implemented |
| `rl/` | 4 | **Trained, disabled in production** | PPO trained for 2M+ steps; three model variants in `models/`. Fusion config sets `WEIGHT_RL=0.0` due to out-of-sample regime mismatch (val Sharpe 1.926, test Sharpe –0.308). Runs every cycle and writes `last_rl_signal.json` but the fusion aggregator ignores it. See `rl/README.md` and the PPO Redesign note for details. |
| Paper trader | 4 | **Operational simulator** | Consumes `last_fused_decision.json`; no real trades. Includes its own pre-trade Python Guardian (`classification/guardian_agent.py`) distinct from the TypeScript Guardian in `services/execution/`. |

---

## Setup

### Requirements

```bash
# Classification + realtime pipeline
pip install xgboost scikit-learn pandas numpy joblib ccxt

# NLP / FinBERT
pip install torch transformers feedparser pydantic pydantic-settings

# Spoofing detection (TEN-GNN)
pip install torch torch-geometric

# Testing
pip install pytest
```

A `requirements.txt` exists in both `classification/` and `nlp/`.

### Environment Variables

All secrets and connection strings must come from environment variables (never hardcoded). Create a `.env` file in `services/models/nlp/` or export them in your shell:

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESCALE_DSN` | empty string | TimescaleDB connection; must be set via env when storage is used |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cache for deduplication |
| `RSS_FEEDS` | 7 feed URLs | Comma-separated list of RSS feed URLs |
| `GDELT_KEYWORDS` | Solana keyword list | Comma-separated GDELT search terms |
| `REDDIT_SUBREDDITS` | `solana,CryptoCurrency,CryptoMarkets` | Comma-separated subreddit list |
| `GDELT_FETCH_INTERVAL_MINUTES` | `60` | GDELT lookback window in minutes |
| `PAPER_TRADER_MAX_SIGNAL_AGE_MINUTES` | `90` | Blocks stale paper-trade entries |
| `SPOOFING_MAX_ORDERBOOK_AGE_MINUTES` | `30` | Blocks stale spoofing scores |

**Important:** Production secrets and DSNs must come from environment variables
or a secrets manager. Do not commit real credentials.

---

## Data Sources

| Source | Module | Cadence | What it provides |
|--------|--------|---------|-----------------|
| Binance REST | `01_collect_sol_data_15m.py`, `realtime_market_fetcher.py`, `realtime_btc_fetcher.py` | 15 min | SOL/USDT + BTC/USDT OHLCV |
| Binance WebSocket | `realtime_orderbook_collector.py` | 1-min aggregation | SOL/USDT partial-depth L2 orderbook features |
| GDELT | `nlp/src/fetchers/gdelt_fetcher.py` | ~60 min | Global event stream filtered for Solana keywords |
| RSS | `nlp/src/fetchers/rss_fetcher.py` | ~15 min | CoinDesk, CoinTelegraph, Decrypt, Solana.com … |
| Reddit | `nlp/src/fetchers/reddit_fetcher.py` | on demand | r/solana, r/CryptoCurrency subreddit posts |
| OpenBook WebSocket | planned | tick | Not connected yet |

---

## How to Collect Data

```bash
cd services/models/classification

# 1. Download historical SOL OHLCV (needs Binance access — no API key required for public data)
python 01_collect_sol_data_15m.py

# 2. Add BTC co-features
python 02b_add_btc_features.py
```

---

## How to Train

```bash
cd services/models/classification

# Build all technical features
python 02_build_technical_features.py

# Merge sentiment features (requires NLP pipeline to have produced processed CSVs)
python 06_merge_sentiment_features.py
python 06b_merge_orderbook_features.py

# Prepare leak-filtered splits (outputs NPY arrays + scalers + feature registry)
python 07_prepare_no_leak_data.py

# Train XGBoost for H4 / H8 / H16
python 08_train_xgboost_no_leak.py
```

### Reproducibility

Training is seeded via `XGB_SEED = 42` in `config.py`. All hyperparameters live in `config.py`. Model artifacts are written to `models/xgboost_h{N}.json`; scalers to `data/processed/scaler_h{N}.pkl`.

---

## How to Evaluate

```bash
cd services/models/classification

# Walk-forward OOS validation (5 expanding folds, purge gap = horizon)
python 20_walk_forward_validation.py

# SL/TP position backtest on held-out test split
python 23_backtest.py

# Confidence gate sweep (optimize P(up) threshold for trading precision)
python 24_gate_sweep.py
```

### Latest Model Performance (H4, retrained 2026-05-10)

| Metric | Value |
|--------|-------|
| Test accuracy | 63.9 % |
| AUC-ROC | 0.617 |
| F1 (@ 0.5) | 0.401 |
| MCC | 0.143 |
| Walk-forward accuracy | ~61 % (5-fold) |
| Backtest win rate | 74.34 % at gate 0.69 |
| Backtest Sharpe | 5.263 at gate 0.69 |
| Backtest max drawdown | 23.26 % at gate 0.69 |
| Confidence threshold | 0.69 (current production gate) |

The earlier `0.62` threshold was the initial walk-forward F1-calibrated gate.
`0.69` is the current configured gate after `24_gate_sweep.py` showed better
Sharpe and drawdown.

---

## How to Run the Realtime Pipeline

The pipeline is designed to be run every 15 minutes, but the orderbook collector
must run continuously in a separate process first:

```bash
cd services/models/classification

# 1. Start this in a long-running terminal/service
python realtime_orderbook_collector.py --symbol solusdt --top-n 10 --flush-interval 60
```

Then run the validated realtime path:

```bash
cd services/models/classification

# 2. Deterministic local validation path (uses cached sentiment)
python run_realtime_pipeline.py --skip-sentiment --fusion --fusion-aggregate --paper-trade --timeout 60

# 3. Full live sentiment path; may need a longer timeout because GDELT/HF can be slow
python run_realtime_pipeline.py --fusion --fusion-aggregate --paper-trade --timeout 120
```

Current step order:

```text
market_fetch → btc_fetch → feature_builder → sentiment_fetch? → orderbook_preflight
→ feature_merge → predictor → dashboard → fusion → spoofing_detect
→ fusion_aggregate → paper_trader
```

The preflight checks `data/realtime/orderbook_status.json`. If the collector is
stopped, disconnected, stale, or missing, the pipeline stops in degraded mode
before XGBoost/fusion/spoofing instead of scoring fake or zero-filled orderbook
features.

Detailed two-terminal runbook:
`classification/run_realtime_orderbook_then_pipeline.md`.

The XGBoost predictor writes `data/realtime/last_signal.json`:

```json
{
  "status": "ok",
  "signal": "DOWN",
  "reason": "CONFIDENT_DOWN",
  "probability_up": 0.2566,
  "probability_down": 0.7434,
  "confidence_threshold": 0.69,
  "horizon": 4,
  "horizon_minutes": 60
}
```

The main downstream contract is `data/realtime/last_fused_decision.json`:

```json
{
  "status": "ok",
  "asset": "SOL/USDT",
  "action": "HOLD",
  "fused_score": -0.1293,
  "risk_level": "LOW",
  "latest_timestamp": "2026-05-13 14:15:00+00:00"
}
```

---

## How to Run Tests

`nlp/src/` and `spoofing_detection/src/` both expose a top-level `src` package.
Python caches whichever is imported first, so they cannot share a pytest session.
Run each suite as a separate invocation:

```bash
cd services/models

# Classification pipeline tests (69 passed locally on 2026-05-13)
python -m pytest classification/tests/ -q

# NLP / FinBERT tests
python -m pytest nlp/tests/ -q

# Fusion + spoofing detection tests
python -m pytest fusion/tests/ spoofing_detection/tests/ -q

# Cross-cutting ML quality invariants
python -m pytest tests/ -q
```

---

## Model Artifacts and Expected Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| XGBoost H4 | `models/xgboost_h4.json` | Production classifier (1-hour horizon) |
| XGBoost H8 | `models/xgboost_h8.json` | 2-hour horizon classifier |
| XGBoost H16 | `models/xgboost_h16.json` | 4-hour horizon classifier |
| Scaler H4 | `classification/data/processed/scaler_h4.pkl` | StandardScaler fitted on train split |
| Feature registry | `classification/data/processed/feature_columns.json` | Ordered list of 47 feature names |
| Calibrated threshold | `classification/data/processed/threshold_h4.json` | Val-F1-optimal threshold |
| Walk-forward results | `models/walk_forward_results.json` | OOS validation metrics |
| Backtest results | `models/backtest_results.json` | SL/TP backtest P&L |
| Gate sweep results | `models/gate_sweep_results.json` | Confidence gate sweep |

---

## Feature Engineering

47 trained features per 15-min candle, grouped as:

| Category | Count | Examples |
|----------|-------|---------|
| Technical | 30 | `rsi_14`, `macd_hist_pct`, `bb_width`, `atr_pct_14`, `volatility_12` |
| Time-of-day | 5 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend` |
| BTC co-features | 5 | `btc_return_4`, `btc_return_16`, `btc_rsi_14`, `btc_vol_12`, `btc_vs_sma20` |
| Orderbook | 7 | `imbalance_mean`, `imbalance_std`, `spread_mean`, `mid_price`, `bid_volume_mean`, `ask_volume_mean`, `pressure_indicator` |

Non-stationary columns (raw price levels, absolute MAs) are excluded by `NON_STATIONARY_COLUMNS` in `config.py`. Future-information columns are blocked by `LEAKAGE_SUBSTRINGS`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FEATURE REGISTRY MISSING` | `feature_columns.json` absent | Run `07_prepare_no_leak_data.py` |
| `MODEL CONTRACT ERROR` | Model/scaler/registry count mismatch | Retrain after feature changes |
| `INSUFFICIENT FEATURES` | Missing trained columns, often stale/absent orderbook columns | Start `realtime_orderbook_collector.py`, wait for fresh rows, rerun pipeline |
| `ORDERBOOK UNAVAILABLE` | Latest candle did not match a fresh orderbook bucket | Keep `realtime_orderbook_collector.py` running continuously |
| `stale_orderbook` | Spoofing detector sees orderbook rows older than `SPOOFING_MAX_ORDERBOOK_AGE_MINUTES` | Restart collector and wait for fresh rows |
| `sentiment_fetch` timeout | GDELT rate limit, invalid JSON, or HuggingFace latency | Use `--skip-sentiment` for local validation or increase `--timeout` |
| `SCALER MISSING` | Scaler pkl not found | Run `07_prepare_no_leak_data.py` |
| `MODEL MISSING` | XGBoost JSON not found | Run `08_train_xgboost_no_leak.py` |
| BTC zero-fill WARNING | `latest_btc_15m.csv` absent | Run `realtime_btc_fetcher.py` |
| FinBERT slow / OOM | No GPU available | Set `CUDA_VISIBLE_DEVICES=""` for CPU inference |
| Walk-forward data error | `sol_usdt_15m_features_h4_with_sentiment_orderbook.csv` missing | Re-run steps 01 → 06b |

---

## Known Limitations

- **No live Solana execution**: The system produces signals and paper-trading decisions only. Wallet integration and the Solana execution layer (Layer 5) are not implemented.
- **Orderbook collector must be supervised**: Fresh XGBoost/DNN/spoofing inference requires `realtime_orderbook_collector.py` to run continuously before the 15-minute pipeline.
- **Live sentiment can timeout**: GDELT rate limits and HuggingFace latency can exceed short pipeline timeouts. The validated local path uses `--skip-sentiment`.
- **Spoofing detection uses synthetic labels**: The TEN-GNN model was trained on synthetically injected spoof patterns using LOBSTER data. It has not been validated against real Solana DEX spoof events.
- **Orderbook features are required by shape, but need retraining validation**: The current model/scaler expects seven orderbook columns. They are now fresh in realtime, but the production model should be retrained/evaluated with a longer live orderbook history.
- **Hawkes MLE is a realtime gate only**: It is implemented in `realtime_spoofing_detector.py`, but not yet validated as a historical training feature.
- **RL, spatial, temporal stubs**: `rl/`, `spatial/`, `temporal/` directories contain only README files.
- **Reddit fetcher unvalidated at scale**: The unauthenticated Reddit JSON endpoint has undocumented rate limits.
