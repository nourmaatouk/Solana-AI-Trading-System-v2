"""
25_fusion_backtest.py — Held-out test set evaluation of the 3-model fusion

Evaluates XGBoost + DNN + TEN (logistic stacker) on the last 15% of labeled
rows (the same test split used by train_dnn_fusion.py), and compares against
XGBoost-only on the identical test period.

Key differences vs 23_backtest.py:
  - Uses pre-trained artifacts (no per-fold retraining) — models were trained
    on the 70% train + 15% val splits; this evaluates the held-out 15%
  - Adds per-candle SL/TP exits (scans high/low between entry and horizon)
  - Adds confidence-based position sizing (MIN_POSITION_SIZE=0.25 at gate)
  - Compares XGB-only vs Fusion on the same test period

Comparable walk-forward baseline (23_backtest.py, XGBoost-only):
  Win rate 74.3%  Sharpe 5.26  1,956 trades  gate=0.69  (fixed-horizon only)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

from config import (
    ACTIVE_HORIZON,
    CONFIDENCE_THRESHOLD,
    LEAKAGE_SUBSTRINGS,
    MIN_POSITION_SIZE,
    MODEL_DIR,
    NON_STATIONARY_COLUMNS,
    ORDERBOOK_CSV,
    PROCESSED_DIR,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAIN_RATIO,
    VAL_RATIO,
)

HORIZON      = ACTIVE_HORIZON
FEE_RT       = 0.001
GATE         = CONFIDENCE_THRESHOLD
RESULTS_PATH = MODEL_DIR / "fusion_backtest_results.json"
INPUT_PATH   = PROCESSED_DIR / ORDERBOOK_CSV

SCALER_PATH  = PROCESSED_DIR / f"scaler_h{HORIZON}.pkl"
XGB_PATH     = MODEL_DIR / f"xgboost_h{HORIZON}.json"
DNN_PATH     = MODEL_DIR / f"dnn_h{HORIZON}.pt"
TEN_PATH     = MODEL_DIR / f"ten_h{HORIZON}.pt"
STACKER_PATH = MODEL_DIR / f"fusion_stacker_h{HORIZON}.pkl"
META_PATH    = MODEL_DIR / f"fusion_meta_h{HORIZON}.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (must match training scripts exactly)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TENClassifier(nn.Module):
    def __init__(self, d_input: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, seq_len: int):
        super().__init__()
        self.embed   = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos_emb[:, :x.size(1)]
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_exit(
    candle_in: int,
    direction: str,
    close_full: np.ndarray,
    high_full: np.ndarray,
    low_full: np.ndarray,
    n_full: int,
) -> tuple[float, str]:
    """Scan candles in [entry+1, entry+HORIZON] for SL/TP; fall back to horizon exit."""
    entry = close_full[candle_in]
    for h in range(1, HORIZON + 1):
        ci = candle_in + h
        if ci >= n_full:
            break
        if direction == "UP":
            if high_full[ci] >= entry * (1.0 + TAKE_PROFIT_PCT):
                return +TAKE_PROFIT_PCT, "TAKE_PROFIT"
            if low_full[ci] <= entry * (1.0 - STOP_LOSS_PCT):
                return -STOP_LOSS_PCT, "STOP_LOSS"
        else:
            if low_full[ci] <= entry * (1.0 - TAKE_PROFIT_PCT):
                return +TAKE_PROFIT_PCT, "TAKE_PROFIT"
            if high_full[ci] >= entry * (1.0 + STOP_LOSS_PCT):
                return -STOP_LOSS_PCT, "STOP_LOSS"
    candle_ex = min(candle_in + HORIZON, n_full - 1)
    raw = (close_full[candle_ex] - entry) / entry
    gross = raw if direction == "UP" else -raw
    return gross, "HORIZON"


def _position_size(prob_up: float, direction: str) -> float:
    excess     = (prob_up - GATE) if direction == "UP" else ((1.0 - GATE) - prob_up)
    max_excess = 1.0 - GATE
    factor     = min(max(excess / max_excess, 0.0), 1.0)
    return MIN_POSITION_SIZE + (1.0 - MIN_POSITION_SIZE) * factor


def _simulate_trades(
    test_proba: np.ndarray,
    orig_idx_test: np.ndarray,
    close_full: np.ndarray,
    high_full: np.ndarray,
    low_full: np.ndarray,
    n_full: int,
    use_sizing: bool = True,
) -> list[dict]:
    """Simulate trades with cooldown, optional SL/TP, optional confidence sizing."""
    trades: list[dict] = []
    last_entry_candle = -HORIZON

    for i, prob in enumerate(test_proba):
        candle_in = orig_idx_test[i]
        if candle_in + HORIZON >= n_full:
            continue
        if candle_in - last_entry_candle < HORIZON:
            continue

        if prob >= GATE:
            direction = "UP"
        elif prob <= 1.0 - GATE:
            direction = "DOWN"
        else:
            continue

        gross, reason = _simulate_exit(
            candle_in, direction, close_full, high_full, low_full, n_full
        )
        size    = _position_size(prob, direction) if use_sizing else 1.0
        net_pnl = (gross - FEE_RT) * size

        trades.append({
            "candle_in":   int(candle_in),
            "direction":   direction,
            "prob":        float(prob),
            "gross_pnl":   float(gross),
            "net_pnl":     float(net_pnl),
            "size":        float(size),
            "exit_reason": reason,
            "won":         int(gross > 0),
        })
        last_entry_candle = candle_in

    return trades


def _compute_stats(trades: list[dict], test_candle_span: int) -> dict:
    if not trades:
        return {}
    n         = len(trades)
    gross_arr = np.array([t["gross_pnl"] for t in trades])
    net_arr   = np.array([t["net_pnl"]   for t in trades])
    wins      = sum(t["won"] for t in trades)
    equity    = np.cumprod(1.0 + net_arr)
    peak = 1.0; max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    sharpe = 0.0
    if net_arr.std() > 0:
        trades_per_year = n / (test_candle_span / (252 * 96))
        sharpe = net_arr.mean() / net_arr.std() * np.sqrt(trades_per_year)

    exit_counts: dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "HORIZON")
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        "n_trades":       n,
        "win_rate":       float(wins / n),
        "avg_gross_pnl":  float(gross_arr.mean()),
        "avg_net_pnl":    float(net_arr.mean()),
        "sum_gross_pnl":  float(gross_arr.sum()),
        "compounded_net": float(equity[-1] - 1),
        "max_drawdown":   float(max_dd),
        "sharpe":         float(sharpe),
        "exit_counts":    exit_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── STEP 1: LOAD + LABEL FULL DATASET ────────────────────────────────────
    print("=" * 90)
    print("FUSION BACKTEST  —  HELD-OUT TEST SET (70/15/15 SPLIT)")
    print("=" * 90)
    print(f"  Confidence gate : P(up) >= {GATE} or <= {1-GATE}")
    print(f"  SL/TP           : SL={STOP_LOSS_PCT*100:.1f}%  TP={TAKE_PROFIT_PCT*100:.1f}%  "
          f"Horizon={HORIZON} bars ({HORIZON*15} min)")
    print(f"  Fee (RT)        : {FEE_RT*100:.1f}%")
    print()

    df_full  = pd.read_csv(INPUT_PATH)
    ts_col   = "datetime" if "datetime" in df_full.columns else "timestamp"
    n_full   = len(df_full)

    close_full = df_full["close"].values.astype(np.float64)
    high_full  = df_full["high"].values.astype(np.float64)
    low_full   = df_full["low"].values.astype(np.float64)
    times_full = df_full[ts_col].values

    future_max_high = df_full["high"].rolling(window=HORIZON).max().shift(-HORIZON)
    future_min_low  = df_full["low"].rolling(window=HORIZON).min().shift(-HORIZON)
    tp_price = df_full["close"] * (1.0 + TAKE_PROFIT_PCT)
    sl_price = df_full["close"] * (1.0 - STOP_LOSS_PCT)

    hit_tp = future_max_high >= tp_price
    hit_sl = future_min_low  <= sl_price

    df_full["target_direction"] = np.nan
    df_full.loc[hit_tp & ~hit_sl, "target_direction"] = 1
    df_full.loc[hit_sl & ~hit_tp, "target_direction"] = 0
    df_full["_orig_idx"] = np.arange(n_full)

    n_before = len(df_full)
    df = df_full.dropna(subset=["target_direction"]).reset_index(drop=True)
    print(f"  Triple-barrier  : kept {len(df):,} / {n_before:,} rows "
          f"(TP={TAKE_PROFIT_PCT*100:.1f}%, SL={STOP_LOSS_PCT*100:.1f}%)")

    _META = {"timestamp", "datetime", "target_direction", "future_return_h", "_orig_idx"}
    feature_columns = [
        col for col in df.columns
        if col not in _META
        and col.lower() not in NON_STATIONARY_COLUMNS
        and not any(sub in col.lower() for sub in LEAKAGE_SUBSTRINGS)
    ]

    X_df     = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    valid    = X_df.notna().all(axis=1)
    X_df     = X_df.loc[valid].reset_index(drop=True)
    orig_idx = df["_orig_idx"].loc[valid].values.astype(int)
    target   = df["target_direction"].loc[valid].astype(float).reset_index(drop=True)

    X_all     = X_df.values.astype(np.float64)
    y_all     = target.values.astype(np.int64)
    n_labeled = len(X_all)
    print(f"  Labeled rows    : {n_labeled:,}  |  Features: {X_all.shape[1]}")

    test_start       = int(n_labeled * (TRAIN_RATIO + VAL_RATIO))
    test_end         = n_labeled
    n_test           = test_end - test_start
    orig_idx_test    = orig_idx[test_start:test_end]
    y_test           = y_all[test_start:test_end]
    test_candle_span = int(orig_idx_test[-1] - orig_idx_test[0]) + 1

    print(f"  Test set        : rows {test_start:,}–{test_end:,}  ({n_test:,} samples)")
    print(f"  Test period     : {times_full[orig_idx_test[0]]}  →  {times_full[orig_idx_test[-1]]}")
    print()

    # ── STEP 2: LOAD ARTIFACTS ────────────────────────────────────────────────
    print("[STEP 2] Loading pre-trained artifacts...")
    for p in (SCALER_PATH, XGB_PATH, DNN_PATH, STACKER_PATH, META_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}\n"
                                    f"  Run train_dnn_fusion.py first.")

    with open(META_PATH) as f:
        meta = json.load(f)

    scaler  = joblib.load(SCALER_PATH)
    stacker = joblib.load(STACKER_PATH)
    use_ten = TEN_PATH.exists()
    print(f"  Scaler     : {SCALER_PATH.name}")
    print(f"  XGBoost    : {XGB_PATH.name}")
    print(f"  DNN        : {DNN_PATH.name}")
    print(f"  TEN        : {TEN_PATH.name}  ({'FOUND' if use_ten else 'NOT FOUND — using 2-model stacker'})")
    print(f"  Stacker    : {STACKER_PATH.name}  ({meta.get('models_used', [])})")
    print()

    # ── STEP 3: SCALE FEATURES + BUILD PREDICTIONS ───────────────────────────
    print("[STEP 3] Computing model predictions on test set...")

    X_all_scaled  = scaler.transform(X_all).astype(np.float32)
    X_test_scaled = X_all_scaled[test_start:test_end]

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_PATH)
    p_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    print(f"  XGBoost done   : mean={p_xgb.mean():.4f}  std={p_xgb.std():.4f}")

    dnn = MLP(meta["in_dim"], tuple(meta["hidden"]), meta["dropout"]).to(DEVICE)
    dnn.load_state_dict(torch.load(DNN_PATH, map_location=DEVICE, weights_only=True))
    dnn.eval()
    with torch.no_grad():
        logits = dnn(torch.tensor(X_test_scaled, device=DEVICE)).cpu().numpy()
    p_dnn = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    print(f"  DNN done       : mean={p_dnn.mean():.4f}  std={p_dnn.std():.4f}")

    if use_ten:
        seq_len = int(meta.get("seq_len", 32))
        n_feat  = X_all_scaled.shape[1]
        X_full_all    = df_full[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        X_full_scaled = scaler.transform(X_full_all).astype(np.float32)
        seq_batch = np.zeros((n_test, seq_len, n_feat), dtype=np.float32)
        for i, raw_ci in enumerate(orig_idx_test):
            start = max(0, raw_ci - seq_len + 1)
            rows  = X_full_scaled[start : raw_ci + 1]
            seq_batch[i, -len(rows):] = rows
        ten = TENClassifier(
            d_input=meta["in_dim"], d_model=meta["ten_d_model"],
            n_heads=meta["ten_n_heads"], n_layers=meta["ten_n_layers"],
            d_ff=meta["ten_d_ff"], dropout=meta["ten_dropout"], seq_len=seq_len,
        ).to(DEVICE)
        ten.load_state_dict(torch.load(TEN_PATH, map_location=DEVICE, weights_only=True))
        ten.eval()
        CHUNK = 512
        logits_ten: list = []
        with torch.no_grad():
            for chunk_start in range(0, n_test, CHUNK):
                chunk = torch.tensor(seq_batch[chunk_start:chunk_start + CHUNK], device=DEVICE)
                logits_ten.append(ten(chunk).cpu().numpy())
        logits_ten_arr = np.concatenate(logits_ten)
        p_ten = (1.0 / (1.0 + np.exp(-logits_ten_arr))).astype(np.float32)
        print(f"  TEN done       : mean={p_ten.mean():.4f}  std={p_ten.std():.4f}")
    else:
        p_ten = None

    S = np.column_stack([p_xgb, p_dnn, p_ten]) if (use_ten and p_ten is not None) else np.column_stack([p_xgb, p_dnn])
    p_fusion = stacker.predict_proba(S)[:, 1].astype(np.float32)
    print(f"  Fusion stacked : mean={p_fusion.mean():.4f}  std={p_fusion.std():.4f}")
    print()

    # ── STEP 4: SIMULATE TRADES ───────────────────────────────────────────────
    print("[STEP 4] Simulating trades...")
    xgb_trades    = _simulate_trades(p_xgb,    orig_idx_test, close_full, high_full, low_full, n_full, use_sizing=False)
    fusion_trades = _simulate_trades(p_fusion, orig_idx_test, close_full, high_full, low_full, n_full, use_sizing=True)
    xgb_stats     = _compute_stats(xgb_trades,    test_candle_span)
    fusion_stats  = _compute_stats(fusion_trades, test_candle_span)

    # ── STEP 5: PRINT COMPARISON ──────────────────────────────────────────────
    def _fmt_cell(stats: dict, key: str, pct: bool = False, decimals: int = 1) -> str:
        if not stats:
            return f"{'—':>12s}"
        v = stats.get(key, 0)
        return f"{v*100:>11.{decimals}f}%" if pct else f"{v:>12.{decimals}f}"

    print("=" * 90)
    print(f"\n{'':30s}  {'XGB-only':>12s}  {'Fusion':>12s}  {'vs 23_backtest XGB':>20s}")
    print(f"  {'─'*85}")

    baseline = {"n_trades": 1956, "win_rate": 0.743, "sharpe": 5.26, "max_drawdown": None}
    table_rows = [
        ("Trades",        "n_trades",       False, 0),
        ("Win rate",      "win_rate",        True, 1),
        ("Avg gross/trd", "avg_gross_pnl",   True, 3),
        ("Avg net/trd",   "avg_net_pnl",     True, 3),
        ("Compounded",    "compounded_net",  True, 2),
        ("Max drawdown",  "max_drawdown",    True, 2),
        ("Sharpe",        "sharpe",         False, 2),
    ]
    for label, key, pct, dec in table_rows:
        xv = _fmt_cell(xgb_stats,    key, pct, dec)
        fv = _fmt_cell(fusion_stats, key, pct, dec)
        bv = "—"
        if key in baseline and baseline[key] is not None:
            bv = f"{baseline[key]*100:.{dec}f}%" if pct else f"{baseline[key]:.{dec}f}"
        print(f"  {label:<28s}  {xv}  {fv}  {bv:>20s}")

    print()
    print("  Exit breakdown:")
    for tag, stats in [("XGB-only", xgb_stats), ("Fusion", fusion_stats)]:
        ec = stats.get("exit_counts", {})
        if ec:
            parts = "  ".join(f"{k}={v}" for k, v in sorted(ec.items()))
            print(f"    {tag:<10s}: {parts}")

    bah_start = int(orig_idx_test[0])
    bah_end   = int(orig_idx_test[-1])
    bah_ret   = (close_full[bah_end] - close_full[bah_start]) / close_full[bah_start] * 100
    print(f"\n  Buy-and-hold (same test period) : {bah_ret:+.2f}%")
    print()
    print("  NOTE: 23_backtest.py numbers use walk-forward + fixed-horizon only.")
    print("        Fusion numbers use saved models + per-candle SL/TP exits.")
    print("=" * 90)

    # ── STEP 6: SAVE RESULTS ─────────────────────────────────────────────────
    results = {
        "test_period": {
            "start":          str(times_full[orig_idx_test[0]]),
            "end":            str(times_full[orig_idx_test[-1]]),
            "n_labeled_rows": int(n_test),
            "n_candles":      int(test_candle_span),
        },
        "gate":               float(GATE),
        "fee_roundtrip":      FEE_RT,
        "horizon_bars":       HORIZON,
        "sl_pct":             float(STOP_LOSS_PCT),
        "tp_pct":             float(TAKE_PROFIT_PCT),
        "models_used":        ["xgb", "dnn", "ten"] if use_ten else ["xgb", "dnn"],
        "xgb_only":           xgb_stats,
        "fusion":             fusion_stats,
        "buy_and_hold_pct":   float(bah_ret),
        "baseline_23_backtest": {
            "n_trades": 1956, "win_rate": 0.743, "sharpe": 5.26,
            "description": "walk-forward 5-fold XGBoost, fixed-horizon exit, no SL/TP",
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {RESULTS_PATH}")
    print("=" * 90)
    print("FUSION BACKTEST COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
