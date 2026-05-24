# API Server

**Status:** Monitoring + control surface for the realtime ML pipeline. Read-mostly with several scoped write endpoints (preferences, scheduler trigger/stop, retrain trigger). No authentication enforced by default; opt-in via `API_KEY`.

`apps/api/main.py` is a FastAPI service that:
- Serves the realtime pipeline outputs (signal, paper-trading, history, scheduler state) over HTTP/JSON.
- Exposes a few write endpoints used by the dashboard (preferences) and by operators (retrain, stop/start scheduler).
- Performs tri-state health checks across Redis, the signal pipeline, and on-disk artifacts (`/deep-health`).

The Node backend at `apps/trading-dashboard/backend` proxies a subset of these endpoints through `/api/ai/*`.

## Running

```bash
pip install -r apps/api/requirements.txt
.venv/bin/uvicorn apps.api.main:app --reload --port 8000
```

The launched process imports from `services/models/classification/config.py` and `services/models/fusion/src/fusion_aggregator.py` at startup, so those modules must be on `sys.path` (handled in `main.py`).

## Endpoints

### Health and metadata

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service info and endpoint list |
| GET | `/health` | Liveness check (always 200 if the process is up) |
| GET | `/deep-health` | Tri-state health (Redis, signal freshness, file artifacts). 200 with `status: ok/warning/fail` |
| GET | `/system` | Aggregated health of all data files (last_fused_decision, sentiment_status, orderbook_status, etc.) |

### Signal and pipeline state

| Method | Path | Description |
| --- | --- | --- |
| GET | `/signal` | Latest fused trading signal (`last_fused_decision.json`, falls back to `last_signal.json`) |
| GET | `/signal/breakdown` | Fused signal + sentiment + spoofing + risk reasoning for the UI |
| GET | `/rl-signal` | Latest PPO RL signal (`last_rl_signal.json`). Returns 404 if RL model not trained |
| GET | `/status` | Last pipeline run summary (`last_pipeline_run.json`) |
| GET | `/scheduler` | Current scheduler state (`scheduler_status.json`) |
| POST | `/scheduler/stop` | Writes a stop-flag file the scheduler reads on the next loop |
| POST | `/scheduler/run` | Spawns a one-shot `run_realtime_pipeline.py` subprocess. Rate-limited to 1/60s |

### Paper trading

| Method | Path | Description |
| --- | --- | --- |
| GET | `/paper-trading` | Paper trading metrics and open position (`paper_trading_log_fusion.json`) |
| GET | `/history` | Last N paper-trading entries (`paper_trading_history.json`) |

### Preferences

| Method | Path | Description |
| --- | --- | --- |
| GET | `/preferences` | Current user preferences (`preferences.json`) |
| POST | `/preferences` | Update preferences (asset, risk_level, trading_mode). Body: `{asset?, risk_level?, trading_mode?}` |

### Backtesting and validation

| Method | Path | Description |
| --- | --- | --- |
| GET | `/backtest` | Latest historical backtest results (`backtest_results.json`) |
| GET | `/walkforward` | Walk-forward validation results (`walk_forward_results.json`) |
| GET | `/accuracy` | Live model accuracy summary (`live_accuracy_summary.json`). 202 if no data yet |
| GET | `/accuracy/log` | Full live-accuracy log (`live_accuracy_log.json`) |
| GET | `/drift` | Feature/prediction drift status (`drift_status.json`), tri-state |

### Retraining

| Method | Path | Description |
| --- | --- | --- |
| GET | `/retrain/status` | Status of the most recent auto-retrain run (`retrain_log.json`) |
| POST | `/retrain/trigger` | Spawns `auto_retrain.py` as a background subprocess |

## Configuration

Set via environment variables (override the defaults in `main.py`):

| Env var | Default | Purpose |
| --- | --- | --- |
| `CORS_ORIGINS` | `http://localhost:5173,5174,3001` | Comma-separated list of allowed origins |
| `API_KEY` | unset (no auth) | When set, all endpoints except `/` and `/health` require header `X-API-Key: <value>` |
| `PIPELINE_RUN_RATE_LIMIT_SEC` | `60` | Minimum seconds between `/scheduler/run` invocations |

The realtime data directory is hardcoded to `services/models/classification/data/realtime/` relative to the repo root — override via `REALTIME_DIR` is not currently wired in `main.py` (only the Node backend honors that).

## What is not implemented yet

- WebSocket push for live signal updates (the Node backend handles this on a separate channel).
- Per-user authentication beyond the optional global `API_KEY`.
- Database-backed state (everything is JSON on disk; Redis is used only by the NLP layer if configured).
- Direct Solana wallet or execution integration — that lives in `services/execution/` and `apps/trading-dashboard/backend/`.

## Tests

```bash
python -m pytest apps/api/tests/ -v
```

These tests cover endpoint shapes against fixture JSON files.
