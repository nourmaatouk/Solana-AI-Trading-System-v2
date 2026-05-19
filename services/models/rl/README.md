# Reinforcement Learning Agent (PPO)

**Status: Fully implemented and trained.**

PPO agent trained on the same 47-feature processed CSV used by XGBoost. Wired into the fusion aggregator with a 30% weight.

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters, paths, and thresholds (imported by other modules) |
| `environment.py` | `TradingEnv(gymnasium.Env)` — 47 features + 3 portfolio state features, Discrete(3) action space |
| `train_ppo.py` | Training script — fits StandardScaler on train split, trains PPO via stable-baselines3 |
| `evaluate.py` | Backtest on test split — prints Sharpe, max drawdown, win rate; saves equity curve PNG |
| `realtime_rl_predictor.py` | Realtime inference — reads latest features CSV, writes `last_rl_signal.json` |

## Model artifacts

The trained model and scaler live at:
```
services/models/models/ppo_trading_agent.zip
services/models/models/rl_scaler.pkl
```

## Running

```bash
# Train (re-run to retrain)
python services/models/rl/train_ppo.py

# Evaluate on test split
python services/models/rl/evaluate.py

# Realtime inference
python services/models/rl/realtime_rl_predictor.py
```

## Output

`services/models/classification/data/realtime/last_rl_signal.json`:
```json
{
  "signal": "BUY",
  "action": 1,
  "confidence": 0.72,
  "probabilities": {"HOLD": 0.15, "BUY": 0.72, "SELL": 0.13},
  "generated_at_utc": "..."
}
```

## Integration

The fusion aggregator reads `last_rl_signal.json` and applies 30% weight. If the model file is absent the weight redistributes to XGBoost automatically.
