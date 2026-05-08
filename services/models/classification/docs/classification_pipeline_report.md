# Classification Pipeline Report

Last updated: 2026-05-07

## Executive Summary

The classification module predicts whether the latest SOL/USDT market setup looks more like an upside breakout or a downside/risk setup over a fixed 15-minute-candle horizon. The current realtime path uses XGBoost as a probability scorer and then applies safety gates before emitting a final signal.

The model should be read as:

```text
Probability UP = how similar the latest feature row is to historical rows labeled as clean upside barrier hits
Probability DOWN = 1 - Probability UP
Final signal = UP, DOWN, or NEUTRAL after data-quality and confidence checks
```

It does not guarantee that the market will move in that direction. A large part of the architecture is designed to say `NEUTRAL` when the data or confidence is not good enough.

## Architecture Overview

```text
Historical path:
  Binance OHLCV
    -> technical features
    -> optional sentiment/orderbook merges
    -> triple-barrier labels
    -> chronological train/val/test split
    -> train-only scaling
    -> XGBoost H4/H8/H16 models

Realtime path:
  latest Binance OHLCV
    -> live technical features
  live sentiment
    -> 15m sentiment row
  live orderbook snapshots
    -> 15m orderbook bucket
  merged live features
    -> feature registry validation
    -> scaler_h4.pkl
    -> xgboost_h4.json
    -> probability score
    -> confidence and data-health gate
    -> final signal
```

## Horizons

The code uses 15-minute candles. Horizon names are candle counts, not hours.

| Horizon | Candles | Time |
| --- | ---: | --- |
| H4 | 4 | 1 hour |
| H8 | 8 | 2 hours |
| H16 | 16 | 4 hours |

The active realtime predictor currently uses H4:

```python
HORIZON = 4
```

That means the active live score is a one-hour horizon score.

## Data Collection

### `01_collect_sol_data_15m.py`

This script uses `ccxt` to fetch Binance SOL/USDT OHLCV candles in batches. It saves:

```text
data/raw/sol_usdt_15m_raw.csv
data/raw/sol_usdt_15m_raw.parquet
data/processed/sol_usdt_15m_direction_h4.csv
data/processed/sol_usdt_15m_direction_h4.parquet
```

The initial `target_direction` produced here is a simple future-close direction target. In the current no-leak pipeline, this is later replaced by the triple-barrier target in `07_prepare_no_leak_data.py`.

## Feature Engineering

### `02_build_technical_features.py`

Builds technical features from OHLCV rows:

- returns: `return_1`, `return_3`, `return_6`, `return_12`, `return_24`
- trend/moving average features: SMA, EMA, close-vs-average distances
- momentum: RSI, MACD, momentum windows
- volatility: rolling standard deviations, ATR
- Bollinger band features
- stationary z-score features: `z_score_72`, `volume_z_score_72`
- volume and candle-shape features

Output:

```text
data/processed/sol_usdt_15m_features_h4.csv
```

### `realtime_feature_builder.py`

Builds the live version of the same technical feature set from:

```text
data/realtime/latest_sol_15m.csv
```

Output:

```text
data/realtime/latest_sol_15m_features.csv
```

The realtime builder must preserve column names used by the training pipeline. The predictor relies on exact names in `feature_columns.json`.

## Sentiment Features

### Historical sentiment

`06_merge_sentiment_features.py` contains helper functions for merging technical features and sentiment by timestamp. The current `main()` path mostly standardizes the technical dataset and writes a downstream-compatible file. If a true historical sentiment file is available, call `merge_features(tech_path, sent_path, output_path)` with explicit paths.

### Realtime sentiment

`realtime_sentiment_fetcher.py` fetches articles from configured NLP sources, runs FinBERT sentiment processing, aggregates the result to a 15-minute bucket, and writes:

```text
data/realtime/latest_sentiment_15m.csv
```

If no fresh articles are found, it uses a decayed fallback row instead of hard-resetting every sentiment feature to zero.

## Orderbook Features

### `services/ingest/connectors/binance_orderbook.py`

The orderbook collector connects to Binance depth WebSocket data and writes aggregated rows to:

```text
services/models/classification/data/realtime/orderbook_features.csv
```

It computes:

- `imbalance_mean`
- `imbalance_std`
- `spread_mean`
- `mid_price`
- `bid_volume_mean`
- `ask_volume_mean`
- `pressure_indicator`

Timestamps are UTC-aware. This matters because market candles are UTC and the merge step buckets orderbook rows into 15-minute UTC intervals.

### `06b_merge_orderbook_features.py`

For historical/replayed training data, this script resamples orderbook rows to 15-minute buckets and merges them onto the market/sentiment dataset.

### `realtime_merge_features.py`

For live inference, this script merges:

```text
latest_sol_15m_features.csv
latest_sentiment_15m.csv
orderbook_features.csv
```

Output:

```text
data/realtime/latest_features_with_sentiment.csv
```

The live merge adds:

```text
orderbook_data_available
```

The predictor uses this flag. If the model expects orderbook features and the latest candle has no matching orderbook bucket, the final signal becomes:

```text
NEUTRAL (ORDERBOOK UNAVAILABLE)
```

## Labeling And Leakage Control

### `07_prepare_no_leak_data.py`

This is the main no-leak data preparation script. It does five important jobs:

1. Builds a triple-barrier target for H4, H8, and H16.
2. Drops rows where neither or both barriers are hit.
3. Removes known leakage columns such as `future`, `target`, `lead`, and `t+1`.
4. Splits chronologically into train/validation/test.
5. Fits `StandardScaler` only on training rows and transforms validation/test rows.

Triple-barrier settings:

```text
take profit: +2.0%
stop loss:   -1.5%
```

Targets:

| Label | Meaning |
| --- | --- |
| `1` | Future window hits take-profit without hitting stop-loss. |
| `0` | Future window hits stop-loss without hitting take-profit. |
| dropped | Both barriers hit or neither barrier hit. |

Outputs:

```text
X_train_h4.npy, y_train_h4.npy
X_val_h4.npy, y_val_h4.npy
X_test_h4.npy, y_test_h4.npy
scaler_h4.pkl

X_train_h8.npy, y_train_h8.npy
...
scaler_h8.pkl

X_train_h16.npy, y_train_h16.npy
...
scaler_h16.pkl

feature_columns.json
```

`feature_columns.json` is the feature contract used by realtime inference.

## Training

### `08_train_xgboost_no_leak.py`

This script trains an XGBoost classifier for H4/H8/H16.

Key behavior:

- loads horizon-specific `.npy` arrays from `data/processed`
- loads feature names from `feature_columns.json`
- computes `scale_pos_weight` from training labels
- uses `binary:logistic`
- uses `logloss`
- uses early stopping
- writes model files to `services/models/models`

Outputs:

```text
services/models/models/xgboost_h4.json
services/models/models/xgboost_h8.json
services/models/models/xgboost_h16.json
services/models/models/horizon_comparison_metrics.json
```

## Realtime Prediction

### `realtime_predictor.py`

The predictor currently loads:

```text
data/processed/feature_columns.json
data/processed/scaler_h4.pkl
services/models/models/xgboost_h4.json
data/realtime/latest_features_with_sentiment.csv
```

It validates:

- feature registry is non-empty,
- live feature file contains every trained feature,
- latest row has no NaN/inf values,
- scaler feature count equals registry feature count,
- model feature count equals registry feature count,
- latest orderbook bucket is available if orderbook features are part of the trained feature set.

Decision rule:

```text
Probability UP >= 0.60  -> UP (CONFIDENT_UP)
Probability UP <= 0.40  -> DOWN (CONFIDENT_DOWN)
Otherwise               -> NEUTRAL (LOW_CONFIDENCE)
```

Other neutral reasons:

```text
NEUTRAL (INSUFFICIENT FEATURES)
NEUTRAL (INVALID FEATURE VALUES)
NEUTRAL (ORDERBOOK UNAVAILABLE)
NEUTRAL (MODEL CONTRACT ERROR)
```

This is intentional. The classifier is a scoring layer, not the final execution engine.

Every predictor run writes a structured handoff file:

```text
services/models/classification/data/realtime/last_signal.json
```

The file includes:

- generated UTC timestamp,
- final signal,
- reason,
- probability up/down when available,
- latest candle metadata,
- horizon,
- confidence threshold,
- model/scaler/feature paths,
- diagnostic details for failed neutral states.

Downstream code should read this JSON file instead of scraping terminal output.

## Realtime Orchestration

### `run_realtime_pipeline.py`

Runs one full 15-minute classification cycle:

```bash
python services/models/classification/run_realtime_pipeline.py
```

Steps:

1. `realtime_market_fetcher.py`
2. `realtime_feature_builder.py`
3. `realtime_sentiment_fetcher.py`
4. `realtime_merge_features.py`
5. `realtime_predictor.py`
6. `22_realtime_classifier_dashboard.py`

The orderbook WebSocket collector is intentionally not inside this runner because it should run continuously:

```bash
python services/ingest/connectors/binance_orderbook.py
```

Orchestrator outputs:

```text
services/models/classification/data/realtime/last_pipeline_run.json
services/models/classification/data/realtime/logs/realtime_pipeline_<run_id>.log
```

Useful flags:

```bash
python services/models/classification/run_realtime_pipeline.py --dry-run
python services/models/classification/run_realtime_pipeline.py --skip-market --skip-sentiment
python services/models/classification/run_realtime_pipeline.py --skip-dashboard
python services/models/classification/run_realtime_pipeline.py --continue-on-error
```

Recommended schedule:

```text
Always running:
  binance_orderbook.py

Every 5 minutes:
  realtime_sentiment_fetcher.py

Every 15 minutes after candle close:
  run_realtime_pipeline.py
```

Example cron:

```cron
*/5 * * * * cd /home/nourmaatouk/Solana-AI-Trading-System && python services/models/classification/realtime_sentiment_fetcher.py >> services/models/classification/data/realtime/logs/sentiment_cron.log 2>&1
1,16,31,46 * * * * cd /home/nourmaatouk/Solana-AI-Trading-System && python services/models/classification/run_realtime_pipeline.py >> services/models/classification/data/realtime/logs/pipeline_cron.log 2>&1
```

## Dashboard

### `22_realtime_classifier_dashboard.py`

Generates a static HTML report for quick visual testing:

```bash
python services/models/classification/22_realtime_classifier_dashboard.py
```

Output:

```text
services/models/classification/reports/realtime_classifier_dashboard.html
```

The dashboard shows:

- latest signal,
- probability up/down,
- confidence gate,
- orderbook freshness,
- number of scored rows,
- probability chart,
- close-price chart,
- recent scored candle table.

Open with:

```bash
xdg-open services/models/classification/reports/realtime_classifier_dashboard.html
```

Or:

```bash
python -m http.server 8080 -d services/models/classification/reports
```

Then visit:

```text
http://localhost:8080/realtime_classifier_dashboard.html
```

## Recommended Realtime Smoke Test

From repo root:

```bash
python services/models/classification/realtime_market_fetcher.py
python services/models/classification/realtime_feature_builder.py
python services/models/classification/realtime_sentiment_fetcher.py
timeout 75s python services/ingest/connectors/binance_orderbook.py
python services/models/classification/run_realtime_pipeline.py --skip-market --skip-sentiment
```

If orderbook data is stale or unavailable, the predictor should return neutral. That is correct behavior.

## Test Coverage

Run:

```bash
python -m pytest services/models/classification/tests
```

The tests cover:

- no-leak preprocessing helper behavior,
- sentiment merge behavior,
- orderbook merge/resampling behavior,
- orderbook imbalance calculation,
- realtime feature builder output validity,
- realtime merge with orderbook availability flags,
- realtime predictor feature alignment and safety gates,
- realtime sentiment fallback rows.

## Operational Contract

Before trusting a realtime signal, all of these must be true:

1. `feature_columns.json`, `scaler_h4.pkl`, and `xgboost_h4.json` agree on feature count.
2. `latest_features_with_sentiment.csv` contains every trained feature.
3. Latest row has no NaN/inf values.
4. Latest row has fresh orderbook if the trained feature set includes orderbook features.
5. Probability clears the confidence gate.
6. A downstream risk/execution layer still approves the trade.

If any condition fails, the correct signal is neutral.

## Known Limitations And Next Steps

- H4 is currently one hour. Rename horizons or document this everywhere to avoid confusion.
- Historical sentiment merging should be upgraded into a full backfill workflow if sentiment is meant to be a training feature.
- Legacy sequence scripts should be moved to a `legacy/` folder or updated to the current flat no-leak architecture.
- Add SHAP analysis for feature explainability.
- Add paper-trading evaluation with fees, slippage, latency, and position sizing.
- Add an integration test that runs the whole realtime path on a tiny fixture dataset.
