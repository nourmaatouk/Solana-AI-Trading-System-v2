"""
realtime_rl_predictor.py — Run the trained PPO agent on the latest realtime features.

Reads : latest_features_with_sentiment.csv  (built by the upstream pipeline steps)
Writes: data/realtime/last_rl_signal.json

Output format:
    {
        "signal":       "BUY" | "SELL" | "NEUTRAL",
        "action":       0 | 1 | 2,
        "confidence":   0.75,
        "probabilities": {"HOLD": 0.10, "BUY": 0.75, "SELL": 0.15},
        "generated_at_utc": "2026-05-18T..."
    }

If the model has not been trained yet (RL_MODEL_PATH does not exist), exits with
status 0 and writes a NEUTRAL signal so the fusion aggregator degrades gracefully.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    FEATURE_COLUMNS_PATH,
    PORTFOLIO_FEATURE_DIM,
    PPO_DEVICE,
    REALTIME_FEATURES_CSV,
    RL_CONFIDENCE_THRESHOLD,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    RL_SIGNAL_PATH,
)

from stable_baselines3 import PPO

_ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
_SIGNAL_MAP   = {0: "NEUTRAL", 1: "BUY", 2: "SELL"}


def _write_signal(signal: dict) -> None:
    RL_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RL_SIGNAL_PATH.write_text(json.dumps(signal, indent=2))


def _neutral(reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    sig = {
        "signal":        "NEUTRAL",
        "action":        0,
        "confidence":    0.0,
        "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
        "generated_at_utc": now,
        "note": reason,
    }
    _write_signal(sig)
    return sig


def main() -> None:
    print("=" * 60)
    print("realtime_rl_predictor.py")
    print("=" * 60)

    # ── Guard: model must exist ────────────────────────────────────────────────
    if not RL_MODEL_PATH.exists():
        sig = _neutral("PPO model not trained yet — run train_ppo.py first")
        print(f"  {sig['note']}")
        print(f"  Signal: NEUTRAL (no model)")
        print("=" * 60)
        return

    if not RL_SCALER_PATH.exists():
        sig = _neutral("rl_scaler.pkl not found — run train_ppo.py first")
        print(f"  {sig['note']}")
        print("=" * 60)
        return

    if not REALTIME_FEATURES_CSV.exists():
        sig = _neutral("Realtime feature CSV not found — run realtime_feature_builder.py first")
        print(f"  {sig['note']}")
        print("=" * 60)
        return

    # ── Load artifacts ─────────────────────────────────────────────────────────
    feature_cols: list[str] = json.loads(FEATURE_COLUMNS_PATH.read_text())
    scaler = joblib.load(RL_SCALER_PATH)
    agent  = PPO.load(str(RL_MODEL_PATH), device=PPO_DEVICE)

    # ── Prepare observation ───────────────────────────────────────────────────
    df = pd.read_csv(REALTIME_FEATURES_CSV)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Fill missing columns (e.g. stale orderbook) with 0 — same degraded-mode
        # behaviour used by the XGBoost and DNN predictors.
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=missing)],
            axis=1,
        )
        print(f"  NOTE: {len(missing)} features missing (orderbook stale?), filled with 0: {missing[:3]}")

    row = df[feature_cols].iloc[[-1]].fillna(0.0).values
    row_scaled = scaler.transform(row).astype(np.float32)

    portfolio_state = np.zeros(PORTFOLIO_FEATURE_DIM, dtype=np.float32)
    obs = np.concatenate([row_scaled[0], portfolio_state]).reshape(1, -1)

    # ── Predict ────────────────────────────────────────────────────────────────
    action_arr, _ = agent.predict(obs, deterministic=True)
    action_idx = int(np.asarray(action_arr).flat[0])

    device = next(agent.policy.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        dist   = agent.policy.get_distribution(obs_tensor)
        probs  = dist.distribution.probs.squeeze().cpu().numpy()

    probs_dict = {
        "HOLD": round(float(probs[0]), 4),
        "BUY":  round(float(probs[1]), 4),
        "SELL": round(float(probs[2]), 4),
    }
    confidence = float(probs[action_idx])

    if confidence < RL_CONFIDENCE_THRESHOLD:
        signal = "NEUTRAL"
    else:
        signal = _SIGNAL_MAP[action_idx]

    result = {
        "signal":           signal,
        "action":           action_idx,
        "confidence":       round(confidence, 4),
        "probabilities":    probs_dict,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_signal(result)

    print(f"  signal={signal}   confidence={confidence:.4f}")
    print(f"  probabilities: HOLD={probs_dict['HOLD']}  BUY={probs_dict['BUY']}  SELL={probs_dict['SELL']}")
    print(f"  Saved → {RL_SIGNAL_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
