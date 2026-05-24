# Implementation Plan — Solana AI Trading System Models

**As of: 2026-05-13**

---

## Current State Summary

### What works end-to-end today

| Component | State |
|-----------|-------|
| Binance OHLCV collection (SOL + BTC, 15 min) | Operational |
| Technical feature engineering (47 trained features) | Operational |
| XGBoost H4/H8/H16 classifiers | Trained, test AUC 0.617 |
| Walk-forward validation (5 folds, purge gap) | Passes |
| SL/TP backtest (`23_backtest.py`) | Operational |
| Confidence gate sweep (`24_gate_sweep.py`) | Operational |
| FinBERT sentiment (GDELT + RSS + Reddit) | Implemented; live GDELT/HF can timeout |
| Binance orderbook WebSocket collector | Operational as a separate long-running process |
| 15-min realtime pipeline (`run_realtime_pipeline.py`) | Operational with fresh orderbook; validated with `--skip-sentiment --fusion --fusion-aggregate --paper-trade` |
| Realtime XGBoost predictor + dashboard | Operational when orderbook collector is fresh |
| XGB+DNN realtime fusion predictor | Operational; uses 2-input stacker unless metadata declares TEN |
| Fusion aggregator (weighted score) | Operational via `--fusion-aggregate` |
| Paper trading simulator | Operational; consumes `last_fused_decision.json` |
| Realtime Hawkes/TEN spoofing gate | Operational but research-grade |

### What does not exist yet

- OpenBook DEX L2/L3 ingestion
- Helius RPC/WebSocket on-chain ingestion
- Historical Hawkes intensity feature fully validated in training data
- GNN spatial encoder (Layer 3 `spatial/`)
- Temporal encoder as a standalone Layer 3 module (`temporal/`)
- RL execution agent: PPO/DQN (Layer 4)
- ElizaOS orchestrator, Ollama Llama 3, ChromaDB memory (Layer 4)
- Solana execution layer (Layer 5)

---

## Gaps Against Target Architecture

| Gap | Impact | Priority |
|-----|--------|----------|
| Orderbook collector must run continuously outside pipeline | Pipeline degrades if the daemon stops | High |
| Live sentiment can exceed short timeouts due GDELT/HF latency | Full live pipeline can fail unless timeout is raised or sentiment is cached/skipped | High |
| Current model should be retrained/evaluated with longer real orderbook history | Current artifacts expect orderbook columns, but orderbook contribution needs validation | Medium |
| Hawkes MLE only used as realtime spoofing gate | Layer 2 historical feature integration incomplete | Medium |
| `spoofing_detection/` trained on synthetic labels only | Not production-safe | Medium |
| `rl/`, `spatial/`, `temporal/` are stubs | Layers 3 and 4 incomplete | Low–Medium |
| Confidence gate validation | Wrong gate in production if stale | **Fixed** — 0.69 is the validated peak-Sharpe gate; summary JSON updated |
| Fusion aggregator wiring | Multi-model fusion not live if absent | **Fixed** — `realtime_fusion_aggregator.py` + `--fusion-aggregate` flag |
| Indicator functions duplicated between `02_build_technical_features.py` and `realtime_feature_builder.py` | Maintenance risk (drift) | **Fixed** — extracted to `classification/indicators.py` |
| `20_walk_forward_validation.py` executes at import time (top-level `print` + `pd.read_csv`) | Not importable or testable | **Fixed** — all I/O inside `main()` |
| `23_backtest.py` no CLI flag for gate | Cannot test alternate gates without editing code | **Fixed** — `--confidence-threshold` added |
| TimescaleDB/Redis defaults contain literal "password" | Security hygiene | **Fixed** — `TIMESCALE_DSN` default is now `""` |
| `nlp/src` + `spoofing_detection/src` namespace collision silently breaks tests | Test reliability | **Fixed** — conftest guard + documented invocations |
| `nlp/` had 5 FinBERT test errors (MockModel missing `.to()` and tensor inputs) | Test suite unreliable | **Fixed** |
| `integration/validate_kaggle_notebook.py` hardcoded `/home/nourmaatouk/...` path | Breaks on other machines | **Fixed** |
| Realtime orderbook path stale/missing columns | XGBoost/dashboard/fusion/spoofing degraded | **Fixed** — Binance partial-depth collector validated with all 7 trained columns |

---

## Priority Roadmap

### Phase 1 — Make Tests and Config Reliable (1–2 weeks)

**Goal:** Zero test failures across all modules; config is the single source of truth.

- [x] Fix `nlp/tests/test_finbert.py` — `MockModel.to()` + `MockTokenizer` returns tensors
- [x] Fix `integration/validate_kaggle_notebook.py` — remove hardcoded absolute path
- [x] Add cross-cutting security + ML quality audit (`tests/test_security_audit.py`)
- [x] Resolve `CONFIDENCE_THRESHOLD` — ran `23_backtest.py` at 0.62 and 0.69; gate sweep evidence confirms **0.69** is peak-Sharpe (5.26 vs 4.09, max-DD 23% vs 33%); `model_performance_summary.json` updated; audit test now passes
- [x] Refactor `20_walk_forward_validation.py` — all I/O/training inside `main()`; `calibrate_threshold` + `gated_metrics` importable
- [x] Add pytest for walk-forward importability (`test_walk_forward_import.py`)
- [x] Extract shared indicators to `classification/indicators.py` + `test_indicators.py`
- [x] Wire fusion aggregator (`realtime_fusion_aggregator.py`, `--fusion-aggregate` flag) + `test_fusion_aggregator.py`
- [x] Fix `TIMESCALE_DSN` default password removed
- [x] Guard `src` namespace collision in top-level conftest

**Acceptance criteria:**  
Separate pytest invocations pass locally. Do not run all model suites in one
pytest process because of the `src` namespace collision.

---

### Phase 2 — Stabilise Data Ingestion and Feature Generation (current priority)

**Goal:** Reliable, reproducible data pipeline from raw OHLCV to training splits.

- [x] Make `realtime_orderbook_collector.py` the active orderbook writer
- [x] Validate Binance WebSocket connection and fresh `orderbook_features.csv`
- [x] Ensure all 7 trained orderbook columns are present:
  `imbalance_mean`, `imbalance_std`, `spread_mean`, `mid_price`,
  `bid_volume_mean`, `ask_volume_mean`, `pressure_indicator`
- [x] Fix orderbook timestamp parsing/staleness guards in merge/spoofing paths
- [x] Add rich `orderbook_status.json` service health contract
- [x] Add realtime pipeline `orderbook_preflight` gate before model scoring
- [x] Add orderbook + pipeline runbook (`classification/run_realtime_orderbook_then_pipeline.md`)
- [x] Validate realtime pipeline with fresh orderbook:
  `--skip-sentiment --fusion --fusion-aggregate --paper-trade --timeout 60`
- [ ] Re-collect a longer orderbook history while the WebSocket collector is live; retrain H4 model with real orderbook features and re-evaluate feature importance
- [ ] Decide whether `services/ingest/connectors/binance_orderbook.py` should be removed or retained as an ingest-layer duplicate
- [ ] Validate/finish `03_hawkes_intensity.py` as a historical Layer 2 feature, separate from the realtime spoofing gate
- [ ] Validate sentiment→market merge schema with a contract test (check that `06_merge_sentiment_features.py` output matches `feature_columns.json`)
- [ ] Add data quality checks at each pipeline stage: no NaN, no Inf, timestamp monotonicity, correct column count
- [ ] Make live sentiment more robust under short runtime budgets: GDELT backoff, cached fallback, or separate sentiment daemon

---

### Phase 3 — Improve Model Training and Evaluation (2–4 weeks)

**Goal:** Better model quality and trustworthy evaluation.

- [ ] Revalidate `config.CONFIDENCE_THRESHOLD = 0.69` after every retrain/gate sweep
- [ ] Retrain H4 model with a longer live orderbook feature history and re-run full evaluation suite
- [ ] Add calibration curve (Brier score, reliability diagram) to `21_xgboost_curves.py`
- [ ] Add slippage-aware Sharpe and max-drawdown to `23_backtest.py`  
  (current backtest uses fee_roundtrip=0.1 % but ignores slippage)
- [ ] Investigate H8 / H16 models — AUC 0.581 / 0.540 is weak; consider dropping H16 from production
- [ ] Add baseline comparison in `08_train_xgboost_no_leak.py` (majority-class, trend-following) so every run reports edge vs naive baselines
- [ ] Validate spoofing detection on real Solana DEX L2 data; replace synthetic labels with real labels from known spoofing incidents if available

---

### Phase 4 — Implement Fusion and Agent Integration (4–8 weeks)

**Goal:** Harden live multi-model signal fusion; start the first RL execution prototype.

- [x] Wire `fusion/src/fusion_aggregator.py` into `run_realtime_pipeline.py`
  via `realtime_fusion_aggregator.py` and `--fusion-aggregate`
- [x] Make `last_fused_decision.json` the downstream decision contract for paper trading
- [x] Make `realtime_fusion_predictor.py` fail safely when TEN metadata is missing
- [ ] Move fusion action thresholds (0.3, 0.7) and weights from `fusion_aggregator.py` into a `fusion/config.py`
- [ ] Build `temporal/` — implement TEN (Temporal Encoder Network) as a standalone module with:
  - LSTM or Transformer encoder over the 47-feature time-series window
  - Train/eval scripts matching the XGBoost pipeline structure
- [ ] Build `spatial/` — implement GNN spatial encoder:
  - Nodes: SOL, BTC, ETH, selected Solana DeFi tokens
  - Edges: return correlation, on-chain liquidity flow
- [ ] Start RL execution agent (`rl/`):
  - Environment: SOL/USDT paper trading environment with SL/TP and position sizing
  - Policy: PPO from stable-baselines3
  - Reward: risk-adjusted return (Sharpe-like) minus transaction costs
  - Observation: fused signal + portfolio state
- [ ] Connect ChromaDB for episodic trade memory (LLM context for ElizaOS)

---

### Phase 5 — Solana Execution, Simulation, and Monitoring (8–12 weeks)

**Goal:** Safe, monitored on-chain execution.

- [ ] Integrate Solana Agent Kit — construct and sign Jupiter swap instructions
- [ ] Add simulation mode: dry-run all trades via mainnet simulation RPC before signing
- [ ] Add pre-trade sanity checks: slippage estimate, liquidity depth, max position size
- [ ] Implement circuit breaker: halt all trading if 24-hour P&L drops below configurable threshold or if drawdown exceeds `config.MAX_DRAWDOWN_PCT`
- [ ] Add Prometheus / Grafana metrics: signal latency, prediction confidence distribution, trade P&L, pipeline step success rate
- [ ] Add alert on pipeline failures (Slack / email) — hook into `run_realtime_pipeline.py` step results
- [ ] Helius RPC WebSocket integration: on-chain volume, whale alerts, DEX liquidity events as Layer 1 data source

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Confidence threshold stale after retraining | Medium | Trades at wrong gate | Audit test `test_confidence_threshold_and_model_summary_agree`; re-run gate sweep after retrain |
| Feature drift: realtime features diverge from training features | Medium | Silent model degradation | `validate_artifact_contract()` in predictor; feature registry test |
| TimescaleDB/Redis default password reaches production | Low | Security incident | Env-var coverage tests; secrets manager |
| Orderbook WebSocket drop → stale data used | Medium | Wrong signal | `ORDERBOOK_STALE_MINUTES` check in merge path and `SPOOFING_MAX_ORDERBOOK_AGE_MINUTES` in spoofing path |
| Orderbook collector not running before cron | High | Pipeline degrades to missing/orderbook-unavailable states | Run collector as supervised service and alert on `orderbook_status.json` age |
| Spoofing model fails on real data (trained synthetic) | High | False positives | Kept non-critical in pipeline (`critical=False`); validate on real data before promoting |
| FinBERT/GDELT latency or rate limiting | High | Full live sentiment step can timeout | Use cached sentiment path for local validation; raise timeout or split sentiment into daemon |
| Walk-forward accuracy decays over time (regime shift) | Medium | Poor live performance | Scheduled monthly retraining; monitor confidence distribution |

---

## Local Test Invocations

`nlp/src` and `spoofing_detection/src` both expose a top-level `src` package, so
they must run in separate pytest invocations to avoid the
`sys.modules['src']` namespace collision.

```bash
# classification
pip install -r services/models/classification/requirements.txt pytest
pytest services/models/classification/tests/ -v

# nlp
pip install -r services/models/nlp/requirements.txt pytest
pytest services/models/nlp/tests/ -v

# fusion + spoofing
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scipy numpy pandas xgboost scikit-learn pytest
pytest services/models/fusion/tests/ services/models/spoofing_detection/tests/ -v

# ML quality invariants
pip install xgboost scikit-learn joblib pydantic pydantic-settings pytest
pytest services/models/tests/ -v
```

Key invariants to verify:
- All test suites pass, with any expected skips/xfails explicitly documented
- `feature_columns.json` matches the feature count of all three XGBoost models
  and the active realtime count is 47
- Scaler dimension matches model input dimension
- No shuffle before chronological split in training scripts
