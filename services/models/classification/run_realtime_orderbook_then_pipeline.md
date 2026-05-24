# Realtime Orderbook + Pipeline Runbook

The realtime model path requires a continuously running orderbook collector.
Start the collector first, wait for at least one flush, then run the pipeline.

## Terminal 1 — Orderbook Collector

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System

python services/models/classification/realtime_orderbook_collector.py \
  --symbol solusdt \
  --top-n 10 \
  --flush-interval 60
```

The collector writes:

- `services/models/classification/data/realtime/orderbook_features.csv`
- `services/models/classification/data/realtime/orderbook_status.json`

The status JSON should show:

```json
{
  "status": "running",
  "connected": true,
  "source": "binance_partial_depth",
  "latest_orderbook_timestamp": "...",
  "n_rows_in_csv": 180,
  "last_error": null
}
```

## Terminal 2 — Realtime Pipeline

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System

python services/models/classification/run_realtime_pipeline.py \
  --skip-sentiment \
  --fusion \
  --fusion-aggregate \
  --paper-trade \
  --timeout 60
```

For full live sentiment, remove `--skip-sentiment` and use a longer timeout:

```bash
python services/models/classification/run_realtime_pipeline.py \
  --fusion \
  --fusion-aggregate \
  --paper-trade \
  --timeout 120
```

## Safety Behavior

Before model scoring, `run_realtime_pipeline.py` checks
`orderbook_status.json`. If the collector is stopped, disconnected, missing, or
stale, the pipeline stops before XGBoost/fusion/spoofing and writes a degraded
reason to `last_pipeline_run.json`.
