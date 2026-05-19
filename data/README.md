# Data Directory

Local development data (gitignored in production).

## Structure

```
data/
  raw/              — Raw CSV downloads (Binance OHLCV, Kaggle LOB)
  processed/        — Engineered feature CSVs used for training
  model_artifacts/
    xgboost/        — Placeholder; actual checkpoint at services/models/classification/checkpoints/
    ppo/            — Placeholder; actual model at services/models/models/ppo_trading_agent.zip
    gnn/            — Placeholder; actual checkpoint at services/models/spoofing_detection/checkpoints/
```

## Notes

- `model_artifacts/` subdirectories are empty placeholders. All trained model files live inside their respective service directories, not here.
- FinBERT weights are not stored locally — they are downloaded from HuggingFace at runtime.
- `data/` is excluded from git. Do not commit raw or processed CSVs.
