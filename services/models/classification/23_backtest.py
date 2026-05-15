"""
23_backtest.py — Walk-forward signal backtest

Same 5-fold expanding-window structure as 20_walk_forward_validation.py.
For each fold's test period, re-generates out-of-sample predictions and
simulates trades using a fixed-horizon exit:

  UP signal  → long:  return = (close[t+H] - close[t]) / close[t]
  DOWN signal → short: return = (close[t] - close[t+H]) / close[t]

Key correctness details:
  - Triple-barrier labels (same as production training)
  - Exit price is close[t + HORIZON] in the ORIGINAL (unfiltered) candle array,
    not the HORIZON-th triple-barrier row — these differ because ~79% of rows
    are dropped during labeling
  - Cooldown: no new entry within HORIZON bars of the previous entry (no
    overlapping positions)

Usage:
  python 23_backtest.py                            # uses config.CONFIDENCE_THRESHOLD
  python 23_backtest.py --confidence-threshold 0.62  # explicit gate sweep
"""

import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import (
    ACTIVE_HORIZON,
    CONFIDENCE_THRESHOLD,
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

INPUT_PATH = PROCESSED_DIR / ORDERBOOK_CSV

N_FOLDS = 5
HORIZON = ACTIVE_HORIZON
FEE_RT  = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward signal backtest.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Confidence gate: signal when P(up) >= threshold or P(up) <= 1-threshold. "
            f"Defaults to config.CONFIDENCE_THRESHOLD ({CONFIDENCE_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = args.confidence_threshold if args.confidence_threshold is not None else CONFIDENCE_THRESHOLD

    # Results are written to a gate-specific file when using a non-default threshold
    if args.confidence_threshold is not None:
        gate_tag = f"_gate{gate:.2f}".replace(".", "p")
        results_path = MODEL_DIR / f"backtest_results{gate_tag}.json"
    else:
        results_path = MODEL_DIR / "backtest_results.json"

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("SIGNAL BACKTEST  —  WALK-FORWARD OUT-OF-SAMPLE")
    print("=" * 90)
    print(f"  Confidence gate : P(up) >= {gate} or <= {1-gate:.4g}")
    print(f"  Exit horizon    : {HORIZON} bars (= {HORIZON * 15} minutes)")
    print(f"  Fee (round-trip): {FEE_RT*100:.1f}%")
    print(f"  Cooldown        : {HORIZON} bars between entries (no overlapping positions)")
    print()

    df_full = pd.read_csv(INPUT_PATH)
    ts_col  = "datetime" if "datetime" in df_full.columns else "timestamp"

    # Full close array — used for correct exit price lookup
    close_full = df_full["close"].values.astype(np.float64)
    times_full = df_full[ts_col].values
    n_full     = len(close_full)

    # --- Triple-barrier labeling (identical to 07_prepare_no_leak_data.py) ------
    future_max_high = df_full["high"].rolling(window=HORIZON).max().shift(-HORIZON)
    future_min_low  = df_full["low"].rolling(window=HORIZON).min().shift(-HORIZON)
    tp_price = df_full["close"] * (1 + TAKE_PROFIT_PCT)
    sl_price = df_full["close"] * (1 - STOP_LOSS_PCT)

    hit_tp = future_max_high >= tp_price
    hit_sl = future_min_low  <= sl_price

    df_full["target_direction"] = np.nan
    df_full.loc[hit_tp & ~hit_sl, "target_direction"] = 1
    df_full.loc[hit_sl & ~hit_tp, "target_direction"] = 0

    # Store original candle index before dropping ambiguous rows
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

    X_df      = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    valid     = X_df.notna().all(axis=1)
    X_df      = X_df.loc[valid].reset_index(drop=True)
    orig_idx  = df["_orig_idx"].loc[valid].values.astype(int)   # candle indices in df_full
    target    = df["target_direction"].loc[valid].astype(float).reset_index(drop=True)

    X_all = X_df.values.astype(np.float64)
    y_all = target.values.astype(np.int64)

    print(f"  Samples   : {len(X_all):,}  |  Features: {X_all.shape[1]}")
    print(f"  Date range: {times_full[orig_idx[0]]}  →  {times_full[orig_idx[-1]]}")
    print()

    n_total   = len(X_all)
    fold_size = n_total // (N_FOLDS + 1)


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: WALK-FORWARD LOOP WITH P&L TRACKING
    # ─────────────────────────────────────────────────────────────────────────────
    print(f"[STEP 2] Running {N_FOLDS}-fold walk-forward backtest...")
    print("=" * 90)

    all_trades:     list[dict] = []
    fold_summaries: list[dict] = []

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

        X_fit  = X_all[:fit_end];          y_fit = y_all[:fit_end]
        X_val  = X_all[val_start:val_end]; y_val = y_all[val_start:val_end]
        X_test = X_all[test_start:test_end]

        scaler   = StandardScaler()
        X_fit_s  = scaler.fit_transform(X_fit)
        X_val_s  = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        n_neg = int((y_fit == 0).sum())
        n_pos = int((y_fit == 1).sum())
        spw   = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            random_state=XGB_SEED,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X_fit_s, y_fit, eval_set=[(X_val_s, y_val)], verbose=False)

        test_proba = model.predict_proba(X_test_s)[:, 1]

        # --- simulate trades with cooldown ----------------------------------------
        fold_trades:      list[dict] = []
        last_entry_candle = -HORIZON  # candle index in df_full of the last entry

        for i, prob in enumerate(test_proba):
            tb_idx    = test_start + i         # triple-barrier row index
            candle_in = orig_idx[tb_idx]       # original candle index (entry)
            candle_ex = candle_in + HORIZON    # original candle index (exit)

            if candle_ex >= n_full:
                continue  # no exit price available

            # Cooldown: skip if within HORIZON raw candles of the last entry
            if candle_in - last_entry_candle < HORIZON:
                continue

            if prob >= gate:
                direction = "UP"
            elif prob <= 1.0 - gate:
                direction = "DOWN"
            else:
                continue

            entry   = close_full[candle_in]
            exit_   = close_full[candle_ex]
            raw_ret = (exit_ - entry) / entry
            pnl     = raw_ret if direction == "UP" else -raw_ret
            net_pnl = pnl - FEE_RT

            fold_trades.append({
                "fold":      fold + 1,
                "candle_in": int(candle_in),
                "time":      str(times_full[candle_in]),
                "direction": direction,
                "prob":      float(prob),
                "entry":     float(entry),
                "exit":      float(exit_),
                "gross_pnl": float(pnl),
                "net_pnl":   float(net_pnl),
                "won":       int(pnl > 0),
            })
            last_entry_candle = candle_in

        all_trades.extend(fold_trades)

        n  = len(fold_trades)
        if n == 0:
            print(f"  Fold {fold+1}: no trades")
            continue

        gross = np.array([t["gross_pnl"] for t in fold_trades])
        net   = np.array([t["net_pnl"]   for t in fold_trades])
        wins  = sum(t["won"] for t in fold_trades)
        n_up  = sum(1 for t in fold_trades if t["direction"] == "UP")
        n_dn  = n - n_up

        period_start = times_full[orig_idx[test_start]]
        period_end   = times_full[orig_idx[min(test_end - 1, n_total - 1)]]
        n_test_candles = orig_idx[min(test_end - 1, n_total - 1)] - orig_idx[test_start] + 1

        print(f"{'─' * 90}")
        print(f"  FOLD {fold+1}  |  {period_start}  →  {period_end}")
        print(f"  Signals   : {n:,}  (UP={n_up}  DOWN={n_dn})  "
              f"trade_rate={n/len(test_proba)*100:.1f}% of labeled rows")
        print(f"  Win rate  : {wins/n*100:.1f}%")
        print(f"  Gross P&L : {gross.mean()*100:+.3f}% avg/trade  |  "
              f"total={gross.sum()*100:+.2f}%")
        print(f"  Net P&L   : {net.mean()*100:+.3f}% avg/trade   |  "
              f"total={net.sum()*100:+.2f}%")
        if gross.std() > 0:
            trades_per_year = n / (n_test_candles / (252 * 96))
            sharpe = gross.mean() / gross.std() * np.sqrt(trades_per_year)
            print(f"  Sharpe    : {sharpe:.2f}  (annualised by trade frequency)")

        fold_summaries.append({
            "fold":          fold + 1,
            "n_trades":      n,
            "n_up":          n_up,
            "n_dn":          n_dn,
            "trade_rate":    float(n / len(test_proba)),
            "win_rate":      float(wins / n),
            "avg_gross_pnl": float(gross.mean()),
            "sum_gross_pnl": float(gross.sum()),
            "avg_net_pnl":   float(net.mean()),
            "sum_net_pnl":   float(net.sum()),
            "std_pnl":       float(gross.std()),
        })

    print("=" * 90)


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: AGGREGATE RESULTS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Aggregate results")
    print("=" * 90)

    if not all_trades:
        print("  No trades generated across any fold.")
    else:
        gross_all = np.array([t["gross_pnl"] for t in all_trades])
        net_all   = np.array([t["net_pnl"]   for t in all_trades])
        wins_all  = sum(t["won"] for t in all_trades)
        n_all     = len(all_trades)
        n_up_all  = sum(1 for t in all_trades if t["direction"] == "UP")
        n_dn_all  = n_all - n_up_all

        equity      = np.cumprod(1 + net_all)
        total_return = float(equity[-1] - 1) * 100
        peak        = 1.0
        max_dd_pct  = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd_pct:
                max_dd_pct = dd
        max_dd_pct *= 100

        # Sharpe by actual trade frequency over the test period (raw candles)
        sharpe = 0.0
        if net_all.std() > 0:
            test_candle_span = orig_idx[-1] - orig_idx[fold_size]
            trades_per_year  = n_all / (test_candle_span / (252 * 96))
            sharpe = net_all.mean() / net_all.std() * np.sqrt(trades_per_year)

        up_pnl = gross_all[[t["direction"] == "UP"  for t in all_trades]]
        dn_pnl = gross_all[[t["direction"] == "DOWN" for t in all_trades]]
        up_wr  = float(np.mean(up_pnl > 0)) if len(up_pnl) > 0 else 0.0
        dn_wr  = float(np.mean(dn_pnl > 0)) if len(dn_pnl) > 0 else 0.0

        print(f"\n  Total trades    : {n_all:,}  (UP={n_up_all}  DOWN={n_dn_all})")
        print(f"  Win rate (all)  : {wins_all/n_all*100:.1f}%")
        print(f"  Win rate UP     : {up_wr*100:.1f}%")
        print(f"  Win rate DOWN   : {dn_wr*100:.1f}%")
        print()
        print(f"  Avg gross P&L / trade : {gross_all.mean()*100:+.3f}%")
        print(f"  Avg net  P&L / trade  : {net_all.mean()*100:+.3f}%  (after {FEE_RT*100:.1f}% RT fee)")
        print(f"  Total gross return    : {gross_all.sum()*100:+.2f}%  (sum of per-trade returns)")
        print(f"  Compounded net return : {total_return:+.2f}%")
        print(f"  Max drawdown          : {max_dd_pct:.2f}%")
        print(f"  Sharpe ratio          : {sharpe:.2f}")
        print()

        bah_start = orig_idx[fold_size]
        bah_end   = orig_idx[-1]
        bah_ret   = (close_full[bah_end] - close_full[bah_start]) / close_full[bah_start] * 100

        print(f"  --- BASELINE ---")
        print(f"  Buy-and-hold return (same test period) : {bah_ret:+.2f}%")
        print(f"  Signal strategy gross return           : {gross_all.sum()*100:+.2f}%")

        print(f"\n  --- INTERPRETATION ---")
        if net_all.mean() > FEE_RT:
            print(f"  Avg net P&L {net_all.mean()*100:+.3f}% > fee {FEE_RT*100:.1f}% → "
                  f"edge survives transaction costs.")
        elif net_all.mean() > 0:
            print(f"  Avg net P&L positive but thin ({net_all.mean()*100:+.3f}%) — "
                  f"real-world slippage likely erases it.")
        else:
            print(f"  Avg net P&L {net_all.mean()*100:+.3f}% → "
                  f"edge does NOT survive transaction costs.")

        if sharpe > 1.0:
            print(f"  Sharpe {sharpe:.2f} > 1.0 → respectable risk-adjusted return.")
        elif sharpe > 0.5:
            print(f"  Sharpe {sharpe:.2f}: modest. Would need portfolio diversification.")
        else:
            print(f"  Sharpe {sharpe:.2f}: poor risk-adjusted return.")

        results = {
            "confidence_gate":        gate,
            "fee_roundtrip":          FEE_RT,
            "horizon_bars":           HORIZON,
            "cooldown_bars":          HORIZON,
            "n_folds":                N_FOLDS,
            "n_total_trades":         n_all,
            "n_up_signals":           n_up_all,
            "n_dn_signals":           n_dn_all,
            "win_rate":               float(wins_all / n_all),
            "win_rate_up":            up_wr,
            "win_rate_dn":            dn_wr,
            "avg_gross_pnl":          float(gross_all.mean()),
            "avg_net_pnl":            float(net_all.mean()),
            "sum_gross_pnl":          float(gross_all.sum()),
            "compounded_net_return":  float(equity[-1] - 1),
            "max_drawdown":           float(max_dd_pct / 100),
            "sharpe":                 float(sharpe),
            "buy_and_hold_pct":       bah_ret,
            "fold_summaries":         fold_summaries,
        }

        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Saved to: {results_path}")

    print("\n" + "=" * 90)
    print("BACKTEST COMPLETE")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
