# Reinforcement Learning Agent (PPO)

**Status: Trained, but disabled in fusion.**

PPO agent trained on the same 49-feature processed CSV used by XGBoost. It is wired into the fusion aggregator, but its fusion weight is currently **disabled** (`WEIGHT_RL = 0.0` in `services/models/fusion/config.py`): validation Sharpe was strong (1.93) but test Sharpe was −0.31 (regime mismatch). The RL signal is still produced and kept as a visible side signal.

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters, paths, and thresholds (imported by other modules) |
| `environment.py` | `TradingEnv(gymnasium.Env)` — 49 market features + portfolio state, Discrete(3) action space |
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

The fusion aggregator reads `last_rl_signal.json`, but its fusion weight is currently `WEIGHT_RL = 0.0` (disabled), so the RL score does not affect the final decision today — it is kept as a visible side signal. Re-enable it by setting a non-zero `WEIGHT_RL` in `services/models/fusion/config.py`.
