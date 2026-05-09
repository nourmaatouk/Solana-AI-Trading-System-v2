# Solana Classification Engine

This folder contains the SOL/USDT classification pipeline. The current production-style path trains an XGBoost classifier on 15-minute candles, technical features, optional sentiment/orderbook features, and a triple-barrier target. At runtime, the model is used as a probability scorer, not as an always-trade bot.

The safest interpretation is:

```text
market + sentiment + orderbook features
  -> XGBoost probability score
  -> feature/data-health checks
  -> UP, DOWN, or NEUTRAL
```

`NEUTRAL` is an expected outcome. The system should refuse to guess when confidence is low, trained features are missing, or live orderbook data is stale.

## Important Horizon Note

All horizons are measured in 15-minute candles:

| Code | Candles | Wall-clock horizon |
| --- | ---: | --- |
| H4 | 4 | 1 hour |
| H8 | 8 | 2 hours |
| H16 | 16 | 4 hours |

So `H4` does not mean four hours in this codebase. For a true four-hour target, use `H16`.

## Main Files

| File | Purpose |
| --- | --- |
| `01_collect_sol_data_15m.py` | Downloads historical SOL/USDT 15m OHLCV candles from Binance. |
| `02_build_technical_features.py` | Builds RSI, MACD, volatility, z-score, trend, volume, and candle-structure features. |
| `06_merge_sentiment_features.py` | Compatibility/helper layer for merging sentiment with technical features. The current main path mostly passes technical data through unless a sentiment file is explicitly supplied to `merge_features`. |
| `06b_merge_orderbook_features.py` | Merges historical/replayed orderbook features onto the training dataset. |
| `07_prepare_no_leak_data.py` | Builds triple-barrier labels, removes leakage columns, splits chronologically, scales using train only, and writes H4/H8/H16 arrays. |
| `08_train_xgboost_no_leak.py` | Trains XGBoost models for H4/H8/H16 and writes model artifacts. |
| `realtime_market_fetcher.py` | Fetches the latest SOL/USDT 15m candles. |
| `realtime_feature_builder.py` | Builds live technical features using the same naming as the training feature builder. |
| `realtime_sentiment_fetcher.py` | Fetches/aggregates live sentiment, with a decayed fallback row. |
| `realtime_merge_features.py` | Merges live technical, sentiment, and orderbook features. Adds `orderbook_data_available`. |
| `realtime_predictor.py` | Loads the H4 scaler/model, validates the feature contract, scores the latest row, and gates the final signal. |
| `run_realtime_pipeline.py` | Runs one full 15-minute realtime classification cycle and writes a run log. |
| `22_realtime_classifier_dashboard.py` | Generates a small static HTML dashboard for testing the classifier visually. |

Full architecture notes are in [docs/classification_pipeline_report.md](docs/classification_pipeline_report.md).

## Historical Training Workflow

Run from the repository root:

```bash
python services/models/classification/01_collect_sol_data_15m.py
python services/models/classification/02_build_technical_features.py
python services/models/classification/06_merge_sentiment_features.py
python services/models/classification/06b_merge_orderbook_features.py
python services/models/classification/07_prepare_no_leak_data.py
python services/models/classification/08_train_xgboost_no_leak.py
```

Expected key outputs:

```text
services/models/classification/data/processed/feature_columns.json
services/models/classification/data/processed/scaler_h4.pkl
services/models/classification/data/processed/scaler_h8.pkl
services/models/classification/data/processed/scaler_h16.pkl
services/models/models/xgboost_h4.json
services/models/models/xgboost_h8.json
services/models/models/xgboost_h16.json
```

`07_prepare_no_leak_data.py` fits the scaler on training rows only, then transforms validation/test rows. This is important for avoiding train/test leakage.

## Target Definition

The current no-leak target is a triple-barrier label:

- `1`: the future window hits a +2.0% take-profit barrier without also hitting the -1.5% stop-loss barrier.
- `0`: the future window hits a -1.5% stop-loss barrier without also hitting the +2.0% take-profit barrier.
- rows where both barriers or neither barrier are hit are dropped from the training set.

This means the model is not asking, "Will price always go up or down?" It is asking, "Does this setup look more like a clean upside breakout or a downside/risk setup over the chosen horizon?"

## Realtime Inference Workflow

The realtime path expects fresh market features and fresh orderbook features.

One-command live cycle:

```bash
python services/models/classification/run_realtime_pipeline.py
```

This runs market fetch, feature build, sentiment fetch, feature merge, predictor, and dashboard refresh once.

Useful debugging variants:

```bash
python services/models/classification/run_realtime_pipeline.py --dry-run
python services/models/classification/run_realtime_pipeline.py --skip-market --skip-sentiment
python services/models/classification/run_realtime_pipeline.py --skip-dashboard
```

Manual equivalent:

```bash
python services/models/classification/realtime_market_fetcher.py
python services/models/classification/realtime_feature_builder.py
python services/models/classification/realtime_sentiment_fetcher.py
python services/models/classification/realtime_merge_features.py
python services/models/classification/realtime_predictor.py
```

For live orderbook collection, run the Binance WebSocket collector separately:

```bash
python services/ingest/connectors/binance_orderbook.py
```

For a short local smoke sample:

```bash
timeout 75s python services/ingest/connectors/binance_orderbook.py
```

`realtime_predictor.py` currently uses:

```text
HORIZON = 4
SCALER = services/models/classification/data/processed/scaler_h4.pkl
MODEL = services/models/models/xgboost_h4.json
CONFIDENCE_THRESHOLD = 0.60
```

Decision gate:

```text
Probability UP >= 0.60  -> UP
Probability UP <= 0.40  -> DOWN
Otherwise               -> NEUTRAL
```

The predictor also returns `NEUTRAL` if:

- trained features are missing from the live feature file,
- the latest row has NaN/inf values,
- model/scaler/feature registry counts do not match,
- the trained model expects orderbook features but the latest orderbook bucket is unavailable.

Each predictor run writes:

```text
services/models/classification/data/realtime/last_signal.json
```

Each orchestrator run writes:

```text
services/models/classification/data/realtime/last_pipeline_run.json
services/models/classification/data/realtime/logs/realtime_pipeline_<run_id>.log
```

Recommended runtime schedule:

```text
Always on:
  python services/ingest/connectors/binance_orderbook.py

Every 5 minutes:
  python services/models/classification/realtime_sentiment_fetcher.py

Every 15 minutes, just after candle close:
  python services/models/classification/run_realtime_pipeline.py
```

Example cron:

```cron
*/5 * * * * cd /home/nourmaatouk/Solana-AI-Trading-System && python services/models/classification/realtime_sentiment_fetcher.py >> services/models/classification/data/realtime/logs/sentiment_cron.log 2>&1
1,16,31,46 * * * * cd /home/nourmaatouk/Solana-AI-Trading-System && python services/models/classification/run_realtime_pipeline.py >> services/models/classification/data/realtime/logs/pipeline_cron.log 2>&1
```

## Dashboard

Generate the small HTML dashboard:

```bash
python services/models/classification/22_realtime_classifier_dashboard.py
```

Open:

```text
services/models/classification/reports/realtime_classifier_dashboard.html
```

If your shell says `Permission denied`, that means you tried to execute the HTML file. Open it with a browser instead:

```bash
xdg-open services/models/classification/reports/realtime_classifier_dashboard.html
```

Or serve it locally:

```bash
python -m http.server 8080 -d services/models/classification/reports
```

Then open:

```text
http://localhost:8080/realtime_classifier_dashboard.html
```

## Testing

Run the classification tests:

```bash
python -m pytest services/models/classification/tests
```

Current expected result after the realtime-path updates:

```text
35 passed
```

## Known Caveats

- The current live predictor uses H4, which is a one-hour horizon because candles are 15 minutes.
- The historical sentiment merge script is not yet a complete historical NLP backfill workflow by itself. Use its helper function with an explicit sentiment CSV if you need a true historical sentiment merge.
- The older sequence scripts (`03_build_sequences.py`, `04_split_data.py`, `05_train_xgboost_baseline.py`, and parts of `21_xgboost_curves.py`) are legacy/compatibility paths and are not the main no-leak H4/H8/H16 pipeline.
- The model output is a signal score, not financial advice and not a guarantee. It still needs risk sizing, execution checks, slippage/fee modeling, and paper trading before real use.
