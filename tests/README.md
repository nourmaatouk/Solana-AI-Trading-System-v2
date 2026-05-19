# Tests

Tests live inside each service directory, not here. The `unit/`, `integration/`, and `e2e/` folders in this directory are empty stubs.

## Where tests actually are

| Location | What is tested |
|----------|---------------|
| `services/models/classification/tests/` | XGBoost pipeline, config, feature engineering |
| `services/models/nlp/tests/` | FinBERT sentiment, NLP contracts, aggregation |
| `services/models/fusion/tests/` | Fusion aggregator, contracts, batch fusion |
| `services/models/spoofing_detection/` | TEN-GNN, Hawkes estimator, benchmarks |
| `services/models/integration/` | End-to-end pipeline tests (13 tests) |
| `apps/api/tests/` | FastAPI endpoint tests |

## Running all tests

```bash
python -m pytest services/models/classification/tests/ -v
python -m pytest services/models/nlp/tests/ -v
python -m pytest services/models/fusion/tests/ -v
python -m pytest services/models/spoofing_detection/ -v
```

CI runs on every push via `.github/workflows/ci.yml`.
