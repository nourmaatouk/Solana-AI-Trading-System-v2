"""
realtime_rl_predictor.py — Run the trained PPO agent on the latest realtime features.

Reads : latest_features_with_sentiment.csv  (built by the upstream pipeline steps)
Writes: data/realtime/last_rl_signal.json
        data/realtime/rl_position_state.json  (persistent position state across calls)

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

Position state persistence:
    The portfolio state [position, unrealised_pnl, steps_since_trade, rel_perf_vs_bh]
    is tracked in rl_position_state.json between pipeline calls.  This mirrors the
    TradingEnv observation exactly so the policy sees the same state distribution it
    was trained on (previously the observation always showed zeros = always flat, which
    biased the policy toward BUY on every single call).
"""
from __future__ import annotations

import json
import math
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
    FEE_RT,
    FEATURE_COLUMNS_PATH,
    FUSION_RISK_HIGH_THRESHOLD,
    FUSION_RISK_MEDIUM_THRESHOLD,
    FUSION_SIGNAL_DIM,
    INITIAL_EQUITY,
    PIPELINE_IDX_FUSED,
    PIPELINE_IDX_RISK,
    PIPELINE_IDX_XGB,
    PIPELINE_SIGNAL_DIM,
    PORTFOLIO_FEATURE_DIM,
    PPO_DEVICE,
    REALTIME_FEATURES_CSV,
    RL_ALIGNED_MODEL_PATH,
    RL_ALIGNED_SCALER_PATH,
    RL_CONFIDENCE_THRESHOLD,
    RL_FUSION_MODEL_PATH,
    RL_FUSION_SCALER_PATH,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    RL_SIGNAL_PATH,
    RL_STATE_PATH,
    SLIPPAGE_PCT,
    TRADE_PENALTY,
    XGB_FUSED_SIGNAL_PATH,
    XGB_SIGNAL_PATH,
)

from stable_baselines3 import PPO

_ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
_SIGNAL_MAP   = {0: "NEUTRAL", 1: "BUY", 2: "SELL"}

# Half round-trip cost applied on entry or exit (matches environment.py)
_COST_HALF = (FEE_RT + SLIPPAGE_PCT) / 2.0


# ── Position state persistence ─────────────────────────────────────────────────

def _load_position_state() -> dict:
    """Load persistent position state from disk. Returns cold-start defaults if absent."""
    defaults: dict = {
        "position":          0.0,
        "entry_price":       0.0,
        "steps_since_trade": 0,
        "equity":            float(INITIAL_EQUITY),
        "bh_equity":         float(INITIAL_EQUITY),
        "peak_equity":       float(INITIAL_EQUITY),
        "last_price":        0.0,
    }
    if not RL_STATE_PATH.exists():
        return defaults
    try:
        saved = json.loads(RL_STATE_PATH.read_text())
        # Fill any keys missing from older versions of the file with defaults.
        for k, v in defaults.items():
            saved.setdefault(k, v)
        return saved
    except Exception:
        return defaults


def _save_position_state(state: dict) -> None:
    RL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RL_STATE_PATH.write_text(json.dumps(state, indent=2))


def _build_portfolio_obs(state: dict, current_price: float) -> np.ndarray:
    """
    Compute the 4-dim portfolio feature vector identical to TradingEnv._get_obs().
    Fields: [position, unrealised_pnl_pct, steps_since_trade/100, log(equity/bh_equity)]
    """
    position = float(state["position"])
    entry_price = float(state["entry_price"])
    equity = float(state["equity"])
    bh_equity = float(state["bh_equity"])

    unrealised_pnl = 0.0
    if position == 1.0 and entry_price > 0.0:
        unrealised_pnl = (current_price - entry_price) / entry_price

    rel_perf = math.log(equity / max(bh_equity, 1e-10))

    return np.array([
        position,
        float(np.clip(unrealised_pnl, -1.0, 1.0)),
        min(float(state["steps_since_trade"]) / 100.0, 1.0),
        float(np.clip(rel_perf, -0.5, 0.5)),
    ], dtype=np.float32)


def _advance_position_state(state: dict, action_idx: int, current_price: float) -> dict:
    """
    Apply action to the position state and return the updated dict.
    Mirrors the cost and equity logic in TradingEnv.step().
    """
    prev_price = float(state["last_price"]) if float(state["last_price"]) > 0.0 else current_price
    prev_position = float(state["position"])

    # ── Mark-to-market for the step ────────────────────────────────────────────
    # Apply the price change that occurred since the last pipeline call.
    if prev_price > 0.0:
        price_ratio = current_price / prev_price
        state["bh_equity"] = float(state["bh_equity"]) * price_ratio
        if prev_position == 1.0:
            state["equity"] = float(state["equity"]) * price_ratio

    # ── Apply action ───────────────────────────────────────────────────────────
    cost = 0.0
    if action_idx == 1 and prev_position == 0.0:       # BUY
        state["position"]    = 1.0
        state["entry_price"] = current_price
        cost = _COST_HALF + TRADE_PENALTY
        state["steps_since_trade"] = 0
    elif action_idx == 2 and prev_position == 1.0:     # SELL
        state["position"]    = 0.0
        state["entry_price"] = 0.0
        cost = _COST_HALF + TRADE_PENALTY
        state["steps_since_trade"] = 0
    else:                                               # HOLD or no-op
        state["steps_since_trade"] = int(state["steps_since_trade"]) + 1

    state["equity"] = float(state["equity"]) * (1.0 - cost)
    state["peak_equity"] = max(float(state["peak_equity"]), float(state["equity"]))
    state["last_price"] = current_price
    return state


# ── Signal helpers ─────────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

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

    # ── Model selection: fusion v2 (56-dim) > aligned (54-dim) > production (53-dim)
    use_fusion  = RL_FUSION_MODEL_PATH.exists()  and RL_FUSION_SCALER_PATH.exists()
    use_aligned = RL_ALIGNED_MODEL_PATH.exists() and RL_ALIGNED_SCALER_PATH.exists()

    # ── Load artifacts ─────────────────────────────────────────────────────────
    feature_cols: list[str] = json.loads(FEATURE_COLUMNS_PATH.read_text())
    if use_fusion:
        scaler     = joblib.load(RL_FUSION_SCALER_PATH)
        agent      = PPO.load(str(RL_FUSION_MODEL_PATH), device=PPO_DEVICE)
        model_name = RL_FUSION_MODEL_PATH.name
        obs_label  = f"{49 + FUSION_SIGNAL_DIM + 4}-dim fusion-v2"
    elif use_aligned:
        scaler     = joblib.load(RL_ALIGNED_SCALER_PATH)
        agent      = PPO.load(str(RL_ALIGNED_MODEL_PATH), device=PPO_DEVICE)
        model_name = RL_ALIGNED_MODEL_PATH.name
        obs_label  = f"{49 + PIPELINE_SIGNAL_DIM + 4}-dim aligned"
    else:
        scaler     = joblib.load(RL_SCALER_PATH)
        agent      = PPO.load(str(RL_MODEL_PATH), device=PPO_DEVICE)
        model_name = RL_MODEL_PATH.name
        obs_label  = "53-dim production"
    print(f"  Model: {model_name}  ({obs_label})")

    # ── Prepare market features ────────────────────────────────────────────────
    df = pd.read_csv(REALTIME_FEATURES_CSV)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=missing)],
            axis=1,
        )
        print(f"  NOTE: {len(missing)} features missing (orderbook stale?), filled with 0: {missing[:3]}")

    row = df[feature_cols].iloc[[-1]].fillna(0.0).values
    row_scaled = scaler.transform(row).astype(np.float32)

    # ── Load persistent position state and build portfolio obs ────────────────
    current_price  = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
    position_state = _load_position_state()
    portfolio_obs  = _build_portfolio_obs(position_state, current_price)

    # ── Build observation vector based on selected model ──────────────────────
    if use_fusion:
        # Read full fusion output from last_fused_decision.json — includes
        # sentiment, GNN, and spoofing contributions computed by the upstream pipeline.
        fused_score  = 0.0    # neutral fallbacks
        xgb_prob_up  = 0.5
        risk_encoded = 0.0    # LOW

        if XGB_FUSED_SIGNAL_PATH.exists():
            try:
                fused_data   = json.loads(XGB_FUSED_SIGNAL_PATH.read_text())
                fused_score  = float(fused_data.get("fused_score", 0.0))
                xgb_prob_up  = float(fused_data.get("classification_probability_up", 0.5))
                risk_str     = str(fused_data.get("risk_level", "LOW")).upper()
                risk_encoded = 1.0 if risk_str == "HIGH" else (0.5 if risk_str == "MEDIUM" else 0.0)
            except Exception:
                pass

        if XGB_SIGNAL_PATH.exists() and xgb_prob_up == 0.5:
            try:
                xgb_sig     = json.loads(XGB_SIGNAL_PATH.read_text())
                xgb_prob_up = float(xgb_sig.get("probability_up", 0.5))
            except Exception:
                pass

        # Pipeline obs matches PIPELINE_IDX_* layout: [xgb_prob_up, fused_score, risk_encoded]
        pipeline_obs = np.array([xgb_prob_up, fused_score, risk_encoded], dtype=np.float32)
        if pipeline_obs.shape[0] != FUSION_SIGNAL_DIM:
            raise ValueError(
                f"pipeline_obs dim mismatch: expected {FUSION_SIGNAL_DIM}, got {pipeline_obs.shape[0]}"
            )
        obs = np.concatenate([row_scaled[0], pipeline_obs, portfolio_obs]).reshape(1, -1)
        print(f"  xgb_prob_up={xgb_prob_up:.4f}  fused_score={fused_score:.4f}  "
              f"risk={'HIGH' if risk_encoded==1.0 else 'MEDIUM' if risk_encoded==0.5 else 'LOW'}")

    elif use_aligned:
        xgb_prob_up = 0.5
        if XGB_SIGNAL_PATH.exists():
            try:
                xgb_sig     = json.loads(XGB_SIGNAL_PATH.read_text())
                xgb_prob_up = float(xgb_sig.get("probability_up", 0.5))
            except Exception:
                pass
        pipeline_obs = np.array([xgb_prob_up], dtype=np.float32)
        obs = np.concatenate([row_scaled[0], pipeline_obs, portfolio_obs]).reshape(1, -1)
        print(f"  xgb_prob_up={xgb_prob_up:.4f}")

    else:
        obs = np.concatenate([row_scaled[0], portfolio_obs]).reshape(1, -1)

    print(f"  position={position_state['position']:.0f}  "
          f"entry_price={position_state['entry_price']:.4f}  "
          f"steps_idle={position_state['steps_since_trade']}  "
          f"equity={position_state['equity']:.2f}")

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

    # ── Persist updated position state ────────────────────────────────────────
    updated_state = _advance_position_state(position_state, action_idx, current_price)
    _save_position_state(updated_state)

    result = {
        "signal":              signal,
        "action":              action_idx,
        "confidence":          round(confidence, 4),
        "probabilities":       probs_dict,
        "position":            float(updated_state["position"]),
        "entry_price":         float(updated_state["entry_price"]),
        "equity":              round(float(updated_state["equity"]), 2),
        "model":               model_name,
        "obs_dim":             obs.shape[1],
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
    }
    _write_signal(result)

    print(f"  signal={signal}   confidence={confidence:.4f}")
    print(f"  probabilities: HOLD={probs_dict['HOLD']}  BUY={probs_dict['BUY']}  SELL={probs_dict['SELL']}")
    print(f"  position after: {updated_state['position']:.0f}  equity: ${updated_state['equity']:.2f}")
    print(f"  Saved → {RL_SIGNAL_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
