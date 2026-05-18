"""
24_gate_sweep.py — Confidence gate threshold sweep

Trains 5 walk-forward folds ONCE, collects out-of-sample probabilities, then
sweeps the gate from 0.55 → 0.75 to find the optimal threshold.

Correctness details (matching 23_backtest.py):
  - Triple-barrier labels (same as production training)
  - Exit price is close[t + HORIZON] in the ORIGINAL unfiltered candle array
  - Cooldown: no new entry within HORIZON raw candles of the previous entry

Updates CONFIDENCE_THRESHOLD in config.py with the best value.
"""

import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import (
    ACTIVE_HORIZON,
    CONFIDENCE_THRESHOLD,
    FEE_RT,
    LEAKAGE_SUBSTRINGS,
    MODEL_DIR,
    NON_STATIONARY_COLUMNS,
    ORDERBOOK_CSV,
    PROCESSED_DIR,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    VAL_RATIO,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SEED,
    XGB_SUBSAMPLE,
)

CONFIG_PATH  = Path(__file__).resolve().parent / "config.py"
RESULTS_PATH = MODEL_DIR / "gate_sweep_results.json"
INPUT_PATH   = PROCESSED_DIR / ORDERBOOK_CSV

N_FOLDS       = 5
HORIZON       = ACTIVE_HORIZON
SWEEP         = np.round(np.arange(0.55, 0.76, 0.01), 2)
MIN_TRADES    = 30   # minimum trades required to be considered viable


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("CONFIDENCE GATE SWEEP")
print("=" * 80)
print(f"  Current gate : {CONFIDENCE_THRESHOLD}")
print(f"  Sweep range  : {SWEEP[0]:.2f} → {SWEEP[-1]:.2f}  (step 0.01)")
print(f"  Fee RT       : {FEE_RT*100:.1f}%")
print(f"  Cooldown     : {HORIZON} raw candles between entries")
print()

df_full = pd.read_csv(INPUT_PATH)

# Full close array for correct exit price lookup
close_full = df_full["close"].values.astype(np.float64)
n_full     = len(close_full)

# --- Triple-barrier labeling -------------------------------------------------
future_max_high = df_full["high"].rolling(window=HORIZON).max().shift(-HORIZON)
future_min_low  = df_full["low"].rolling(window=HORIZON).min().shift(-HORIZON)
tp_price = df_full["close"] * (1 + TAKE_PROFIT_PCT)
sl_price = df_full["close"] * (1 - STOP_LOSS_PCT)

hit_tp = future_max_high >= tp_price
hit_sl = future_min_low  <= sl_price

df_full["target_direction"] = np.nan
df_full.loc[hit_tp & ~hit_sl, "target_direction"] = 1
df_full.loc[hit_sl & ~hit_tp, "target_direction"] = 0
df_full["_orig_idx"] = np.arange(n_full)

n_before = len(df_full)
df = df_full.dropna(subset=["target_direction"]).reset_index(drop=True)
print(f"  Triple-barrier: kept {len(df):,} / {n_before:,} rows "
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
n_total   = len(X_all)
fold_size = n_total // (N_FOLDS + 1)

print(f"  Samples  : {n_total:,}  |  Features: {X_all.shape[1]}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FOLDS ONCE — collect (proba, candle_in, candle_ex) tuples
# ─────────────────────────────────────────────────────────────────────────────
print("Training 5 folds (once)...")
print("-" * 80)

# Store per-prediction: probability, raw candle entry index, raw candle exit index
pred_proba:     list[float] = []
pred_candle_in: list[int]   = []
pred_candle_ex: list[int]   = []

for fold in range(N_FOLDS):
    train_end  = fold_size * (fold + 1)
    test_start = train_end
    test_end   = fold_size * (fold + 2)

    val_size  = max(int(VAL_RATIO * train_end), 1)
    val_start = train_end - val_size
    fit_end   = val_start - HORIZON
    val_end   = train_end - HORIZON

    if fit_end <= 0 or val_end <= val_start:
        continue

    X_fit = X_all[:fit_end];          y_fit = y_all[:fit_end]
    X_val = X_all[val_start:val_end]; y_val = y_all[val_start:val_end]
    X_test = X_all[test_start:test_end]

    scaler   = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_val_s  = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    n_neg = int((y_fit == 0).sum())
    n_pos = int((y_fit == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=XGB_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_fit_s, y_fit, eval_set=[(X_val_s, y_val)], verbose=False)

    proba = model.predict_proba(X_test_s)[:, 1]

    for i, p in enumerate(proba):
        tb_idx    = test_start + i
        candle_in = orig_idx[tb_idx]
        candle_ex = candle_in + HORIZON
        if candle_ex >= n_full:
            continue
        pred_proba.append(float(p))
        pred_candle_in.append(int(candle_in))
        pred_candle_ex.append(int(candle_ex))

    print(f"  Fold {fold+1}: {len(proba):,} test predictions  "
          f"(best_iter={model.best_iteration})")

pred_proba     = np.array(pred_proba)
pred_candle_in = np.array(pred_candle_in, dtype=int)
pred_candle_ex = np.array(pred_candle_ex, dtype=int)

print(f"\n  Total predictions : {len(pred_proba):,}")
print()

# Precompute raw returns for every prediction
raw_ret_all = (close_full[pred_candle_ex] - close_full[pred_candle_in]) / close_full[pred_candle_in]

test_candle_span = int(pred_candle_in[-1] - pred_candle_in[0]) + 1
bars_per_year    = 252 * 96


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP THRESHOLDS — apply cooldown during simulation
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print(f"{'Gate':>6}  {'Trades':>7}  {'Rate%':>6}  {'WinRate':>8}  "
      f"{'GrossAvg%':>10}  {'NetAvg%':>9}  {'Sharpe':>7}  {'MaxDD%':>8}")
print("-" * 80)

rows = []

for gate in SWEEP:
    lo = 1.0 - gate

    # Apply cooldown: iterate chronologically, skip signals within HORIZON
    # raw candles of the previous accepted entry
    pnl_list: list[float] = []
    last_candle_in = -HORIZON

    for i in range(len(pred_proba)):
        p         = pred_proba[i]
        candle_in = pred_candle_in[i]

        if p >= gate:
            direction = 1      # UP: long
        elif p <= lo:
            direction = -1     # DOWN: short
        else:
            continue           # neutral

        if candle_in - last_candle_in < HORIZON:
            continue           # cooldown: position still open

        gross   = raw_ret_all[i] * direction
        pnl_list.append(gross)
        last_candle_in = candle_in

    n = len(pnl_list)
    if n < MIN_TRADES:
        continue

    pnl     = np.array(pnl_list)
    net_pnl = pnl - FEE_RT
    wins    = (pnl > 0).sum()
    win_rate = wins / n

    equity = np.cumprod(1 + net_pnl)
    peak   = 1.0
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    tpy    = n / (test_candle_span / bars_per_year)
    sharpe = (net_pnl.mean() / net_pnl.std() * np.sqrt(tpy)) if net_pnl.std() > 0 else 0.0

    rows.append({
        "gate":       float(gate),
        "n_trades":   int(n),
        "trade_rate": float(n / len(pred_proba)),
        "win_rate":   float(win_rate),
        "avg_gross":  float(pnl.mean()),
        "avg_net":    float(net_pnl.mean()),
        "sharpe":     float(sharpe),
        "max_dd":     float(max_dd),
        "is_current": bool(abs(gate - CONFIDENCE_THRESHOLD) < 0.001),
    })

    marker = " ◄ current" if abs(gate - CONFIDENCE_THRESHOLD) < 0.001 else ""
    print(f"  {gate:.2f}  {n:>7,}  {n/len(pred_proba)*100:>6.1f}%  "
          f"{win_rate*100:>7.1f}%  {pnl.mean()*100:>+9.3f}%  "
          f"{net_pnl.mean()*100:>+8.3f}%  {sharpe:>7.2f}  "
          f"{max_dd*100:>7.2f}%{marker}")

print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# PICK BEST — maximise Sharpe among gates where net avg > fee
# ─────────────────────────────────────────────────────────────────────────────
viable = [r for r in rows if r["avg_net"] > FEE_RT]
if not viable:
    viable = [r for r in rows if r["avg_net"] > 0]

best = max(viable, key=lambda r: r["sharpe"]) if viable else None

if best:
    print(f"\n  Best gate : {best['gate']:.2f}  "
          f"(Sharpe={best['sharpe']:.2f}, "
          f"win_rate={best['win_rate']*100:.1f}%, "
          f"net_avg={best['avg_net']*100:+.3f}%, "
          f"n_trades={best['n_trades']:,})")
    print(f"  Current   : {CONFIDENCE_THRESHOLD:.2f}")

    if abs(best["gate"] - CONFIDENCE_THRESHOLD) < 0.001:
        print("  Current gate is already optimal — no change needed.")
        update_config = False
    else:
        update_config = True
else:
    print("  No viable gate found.")
    update_config = False


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE config.py
# ─────────────────────────────────────────────────────────────────────────────
if update_config and best:
    text     = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(CONFIDENCE_THRESHOLD\s*=\s*)\d+\.\d+",
        f"\\g<1>{best['gate']:.2f}",
        text,
    )
    if new_text != text:
        CONFIG_PATH.write_text(new_text)
        print(f"\n  config.py updated: CONFIDENCE_THRESHOLD = {best['gate']:.2f}")
    else:
        print("\n  WARNING: Could not find CONFIDENCE_THRESHOLD in config.py — update manually.")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "sweep":         rows,
    "best":          best,
    "previous_gate": float(CONFIDENCE_THRESHOLD),
    "fee_rt":        FEE_RT,
    "cooldown_bars": HORIZON,
}
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to: {RESULTS_PATH}")
print("\n" + "=" * 80)
print("GATE SWEEP COMPLETE")
print("=" * 80)
