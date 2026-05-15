# API Server

**Status: Minimal implementation — read-only monitoring endpoints only.**

`apps/api/main.py` exposes a FastAPI backend that serves the realtime pipeline
outputs over HTTP.  No authentication, no writes to the model pipeline.

## Running

```bash
pip install -r apps/api/requirements.txt
uvicorn apps.api.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint list |
| GET | `/health` | Liveness check |
| GET | `/signal` | Latest fused trading signal (`last_fused_decision.json`) |
| GET | `/status` | Last pipeline run summary (`last_pipeline_run.json`) |
| GET | `/paper-trading` | Paper trading metrics and open position |
| GET | `/preferences` | Current user preferences |
| POST | `/preferences` | Update preferences (asset, risk_level, trading_mode, etc.) |
| GET | `/system` | Aggregated health of all data files |

## What is not implemented yet

- Authentication and authorization
- WebSocket push for live signal updates
- Rate limiting
- Endpoints for controlling the pipeline (start/stop/retrain)
- Multi-user support
- Integration with Solana wallet or execution layer

## Tests

```bash
python -m pytest apps/api/tests/ -v
```
