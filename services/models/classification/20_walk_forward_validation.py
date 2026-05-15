"""
20_walk_forward_validation.py

WALK-FORWARD TEST — TRUE OUT-OF-SAMPLE VERIFICATION

Methodology:
  1. Expanding training window; each fold tests on a strictly future period.
  2. Per-fold val set (last 20% of training data) for early stopping.
  3. Purge gap (= HORIZON bars) at every train/val and val/test boundary so
     that forward-looking labels never bleed across splits.
  4. Production-matched hyperparameters (from config.py).
  5. Confidence-gated trading metrics: only count signals where
     P(up) >= CONFIDENCE_THRESHOLD or P(up) <= 1-CONFIDENCE_THRESHOLD.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SEED,
    XGB_SUBSAMPLE,
)
from training_utils import calibrate_threshold

INPUT_PATH   = PROCESSED_DIR / ORDERBOOK_CSV
RESULTS_PATH = MODEL_DIR / "walk_forward_results.json"

N_FOLDS    = 5
HORIZON    = ACTIVE_HORIZON  # bars ahead (matches production); also used as purge gap
VAL_RATIO  = 0.20            # fraction of training data held out for early stopping



def gated_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """
    Return trading-relevant metrics for confident predictions only.
    A signal is issued when P(up) >= threshold (UP) or P(up) <= 1-threshold (DOWN).
    Precision for UP = fraction of UP signals that were actually UP.
    Precision for DOWN = fraction of DOWN signals that were actually DOWN.
    """
    up_mask = proba >= threshold
    dn_mask = proba <= 1.0 - threshold
    trade_mask = up_mask | dn_mask

    if trade_mask.sum() == 0:
        return {"trade_rate": 0.0, "n_trades": 0, "precision_up": None,
                "precision_dn": None, "combined_precision": None,
                "n_up": 0, "n_dn": 0}

    prec_up = float(y_true[up_mask].mean()) if up_mask.sum() > 0 else None
    prec_dn = float(1 - y_true[dn_mask].mean()) if dn_mask.sum() > 0 else None

    correct = np.concatenate([
        y_true[up_mask],
        (1 - y_true[dn_mask]),
    ])
    combined = float(correct.mean())

    return {
        "trade_rate":         float(trade_mask.sum() / len(y_true)),
        "n_trades":           int(trade_mask.sum()),
        "n_up":               int(up_mask.sum()),
        "n_dn":               int(dn_mask.sum()),
        "precision_up":       prec_up,
        "precision_dn":       prec_dn,
        "combined_precision": combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 90)
    print("WALK-FORWARD VALIDATION  —  TRUE OUT-OF-SAMPLE TEST")
    print("=" * 90)
    print(f"  Folds              : {N_FOLDS} (expanding window)")
    print(f"  Val ratio          : {VAL_RATIO:.0%} of each fold's training data")
    print(f"  Purge gap          : {HORIZON} bars at every boundary")
    print(f"  Confidence gate    : P(up) >= {CONFIDENCE_THRESHOLD} or <= {1-CONFIDENCE_THRESHOLD}")
    print(f"  Hyperparameters    : production (LR={XGB_LEARNING_RATE}, depth={XGB_MAX_DEPTH}, "
          f"n_est={XGB_N_ESTIMATORS}, ES={XGB_EARLY_STOPPING_ROUNDS})")
    print()


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: LOAD & ALIGN DATA
    # ─────────────────────────────────────────────────────────────────────────────
    print("[STEP 1] Loading data...")

    df = pd.read_csv(INPUT_PATH)

    # --- Triple-barrier labeling (identical to 07_prepare_no_leak_data.py) ------
    future_max_high = df["high"].rolling(window=HORIZON).max().shift(-HORIZON)
    future_min_low  = df["low"].rolling(window=HORIZON).min().shift(-HORIZON)
    tp_price = df["close"] * (1 + TAKE_PROFIT_PCT)
    sl_price = df["close"] * (1 - STOP_LOSS_PCT)

    hit_tp = future_max_high >= tp_price
    hit_sl = future_min_low  <= sl_price

    df["target_direction"] = np.nan
    df.loc[hit_tp & ~hit_sl, "target_direction"] = 1
    df.loc[hit_sl & ~hit_tp, "target_direction"] = 0
    n_before = len(df)
    df = df.dropna(subset=["target_direction"]).reset_index(drop=True)
    print(f"  Triple-barrier: kept {len(df):,} / {n_before:,} rows "
          f"(TP={TAKE_PROFIT_PCT*100:.1f}%, SL={STOP_LOSS_PCT*100:.1f}%)")

    _META = {"timestamp", "datetime", "target_direction", "future_return_h"}
    feature_columns = [
        col for col in df.columns
        if col not in _META
        and col.lower() not in NON_STATIONARY_COLUMNS
        and not any(sub in col.lower() for sub in LEAKAGE_SUBSTRINGS)
    ]

    X_df = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    valid_mask = X_df.notna().all(axis=1)
    X_df   = X_df.loc[valid_mask].reset_index(drop=True)
    target = df["target_direction"].loc[valid_mask].astype(float).reset_index(drop=True)

    X_all = X_df.values.astype(np.float64)
    y_all = target.values.astype(np.int64)

    print(f"  Samples  : {len(X_all):,}  |  Features : {X_all.shape[1]}")
    print(f"  Class balance — UP: {y_all.mean()*100:.1f}%  DOWN: {(1-y_all).mean()*100:.1f}%")
    print()


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: WALK-FORWARD LOOP
    # ─────────────────────────────────────────────────────────────────────────────
    n_total   = len(X_all)
    fold_size = n_total // (N_FOLDS + 1)

    print(f"[STEP 2] Walk-forward validation  ({N_FOLDS} folds, ~{fold_size:,} rows each)")
    print("=" * 90)

    fold_results: list[dict] = []
    all_true:  list[np.ndarray] = []
    all_proba: list[np.ndarray] = []

    for fold in range(N_FOLDS):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = fold_size * (fold + 2)

        # --- inner val split (chronological, purged) ----------------------------
        val_size  = max(int(VAL_RATIO * train_end), 1)
        val_start = train_end - val_size
        fit_end   = val_start - HORIZON   # purge: last HORIZON fit labels use val-period prices
        val_end   = train_end - HORIZON   # purge: last HORIZON val labels use test-period prices

        if fit_end <= 0 or val_end <= val_start:
            print(f"  Fold {fold+1}: skipped (insufficient data for purge gaps)")
            continue

        X_fit = X_all[:fit_end]
        y_fit = y_all[:fit_end]
        X_val = X_all[val_start:val_end]
        y_val = y_all[val_start:val_end]
        X_test = X_all[test_start:test_end]
        y_test = y_all[test_start:test_end]

        # --- per-fold scaler (fit on training data only) ------------------------
        scaler = StandardScaler()
        X_fit_s  = scaler.fit_transform(X_fit)
        X_val_s  = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # --- train with early stopping (production hyperparameters) ---------------
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
        model.fit(
            X_fit_s, y_fit,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )
        best_iter = model.best_iteration

        # --- calibrate threshold on val set -------------------------------------
        val_proba  = model.predict_proba(X_val_s)[:, 1]
        cal_thresh = calibrate_threshold(y_val, val_proba)

        # --- evaluate on test set ------------------------------------------------
        test_proba = model.predict_proba(X_test_s)[:, 1]
        test_pred  = (test_proba >= 0.5).astype(int)

        acc  = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec  = recall_score(y_test, test_pred, zero_division=0)
        f1   = f1_score(y_test, test_pred, zero_division=0)
        auc  = roc_auc_score(y_test, test_proba)
        mcc  = matthews_corrcoef(y_test, test_pred)

        gated_prod = gated_metrics(y_test, test_proba, CONFIDENCE_THRESHOLD)
        gated_cal  = gated_metrics(y_test, test_proba, cal_thresh)

        print(f"{'─' * 90}")
        print(f"FOLD {fold+1}/{N_FOLDS}  |  fit={len(X_fit):,}  val={len(X_val):,}  "
              f"test={len(X_test):,}  best_iter={best_iter}")
        print(f"  Accuracy : {acc*100:5.2f}%   AUC : {auc:.4f}   MCC : {mcc:.4f}")
        print(f"  F1@0.50  : {f1*100:5.2f}%   Prec: {prec*100:5.2f}%   Recall: {rec*100:5.2f}%")
        print(f"  Cal threshold (val F1) : {cal_thresh:.3f}")
        print(f"  Gate @ {CONFIDENCE_THRESHOLD:.2f} (production) :")
        print(f"    trade_rate={gated_prod['trade_rate']*100:.1f}%  "
              f"n_trades={gated_prod['n_trades']}  "
              f"combined_prec={gated_prod['combined_precision']:.3f}"
              if gated_prod['n_trades'] > 0 else "    (no trades)")
        print(f"  Gate @ {cal_thresh:.3f} (calibrated)  :")
        print(f"    trade_rate={gated_cal['trade_rate']*100:.1f}%  "
              f"n_trades={gated_cal['n_trades']}  "
              f"combined_prec={gated_cal['combined_precision']:.3f}"
              if gated_cal['n_trades'] > 0 else "    (no trades)")

        fold_results.append({
            "fold":           fold + 1,
            "n_fit":          int(len(X_fit)),
            "n_val":          int(len(X_val)),
            "n_test":         int(len(X_test)),
            "best_iteration": int(best_iter),
            "cal_threshold":  float(cal_thresh),
            "accuracy":       float(acc),
            "auc":            float(auc),
            "f1":             float(f1),
            "mcc":            float(mcc),
            "precision":      float(prec),
            "recall":         float(rec),
            "gated_prod":     gated_prod,
            "gated_cal":      gated_cal,
        })
        all_true.append(y_test)
        all_proba.append(test_proba)

    print("=" * 90)


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: AGGREGATE
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Aggregate results")
    print("=" * 90)

    y_all_test  = np.concatenate(all_true)
    p_all_test  = np.concatenate(all_proba)
    pred_all    = (p_all_test >= 0.5).astype(int)

    oa_acc  = accuracy_score(y_all_test, pred_all)
    oa_auc  = roc_auc_score(y_all_test, p_all_test)
    oa_f1   = f1_score(y_all_test, pred_all, zero_division=0)
    oa_mcc  = matthews_corrcoef(y_all_test, pred_all)
    oa_prec = precision_score(y_all_test, pred_all, zero_division=0)
    oa_rec  = recall_score(y_all_test, pred_all, zero_division=0)

    mean_acc = float(np.mean([r["accuracy"] for r in fold_results]))
    std_acc  = float(np.std([r["accuracy"] for r in fold_results]))
    mean_auc = float(np.mean([r["auc"] for r in fold_results]))
    mean_mcc = float(np.mean([r["mcc"] for r in fold_results]))

    baseline_rate = float(y_all_test.mean())
    baseline_acc  = max(baseline_rate, 1 - baseline_rate) * 100

    gated_oa_prod = gated_metrics(y_all_test, p_all_test, CONFIDENCE_THRESHOLD)
    mean_cal_thr  = float(np.mean([r["cal_threshold"] for r in fold_results]))
    gated_oa_cal  = gated_metrics(y_all_test, p_all_test, mean_cal_thr)

    fold_acc_str = "  ".join(f"{r['accuracy']*100:.1f}%" for r in fold_results)
    print(f"\n  Per-fold accuracy : {fold_acc_str}")
    print(f"  Mean accuracy     : {mean_acc*100:.2f}%  ±{std_acc*100:.2f}%")
    print(f"  Overall accuracy  : {oa_acc*100:.2f}%   (random baseline: {baseline_acc:.2f}%)")
    print(f"  Overall AUC       : {oa_auc:.4f}       (mean per-fold: {mean_auc:.4f})")
    print(f"  Overall F1        : {oa_f1:.4f}")
    print(f"  Overall MCC       : {oa_mcc:.4f}       (mean per-fold: {mean_mcc:.4f})")
    print(f"  Overall Precision : {oa_prec:.4f}   Recall: {oa_rec:.4f}")

    print(f"\n  --- TRADING EDGE  (confidence gate = {CONFIDENCE_THRESHOLD}) ---")
    if gated_oa_prod["n_trades"] > 0:
        print(f"  Trade rate        : {gated_oa_prod['trade_rate']*100:.1f}%  ({gated_oa_prod['n_trades']:,} signals)")
        print(f"  UP signals        : {gated_oa_prod['n_up']:,}   precision={gated_oa_prod['precision_up']:.3f}")
        print(f"  DOWN signals      : {gated_oa_prod['n_dn']:,}   precision={gated_oa_prod['precision_dn']:.3f}")
        print(f"  Combined precision: {gated_oa_prod['combined_precision']:.3f}  "
              f"(random baseline = {max(baseline_rate, 1-baseline_rate):.3f})")
        edge = gated_oa_prod["combined_precision"] - max(baseline_rate, 1 - baseline_rate)
        print(f"  Edge vs baseline  : {edge:+.3f}")
    else:
        print("  No trades triggered at this threshold.")

    print(f"\n  --- TRADING EDGE  (calibrated gate = {mean_cal_thr:.3f}) ---")
    if gated_oa_cal["n_trades"] > 0:
        print(f"  Trade rate        : {gated_oa_cal['trade_rate']*100:.1f}%  ({gated_oa_cal['n_trades']:,} signals)")
        print(f"  UP signals        : {gated_oa_cal['n_up']:,}   precision={gated_oa_cal['precision_up']:.3f}")
        print(f"  DOWN signals      : {gated_oa_cal['n_dn']:,}   precision={gated_oa_cal['precision_dn']:.3f}")
        print(f"  Combined precision: {gated_oa_cal['combined_precision']:.3f}")
    else:
        print("  No trades triggered.")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4: INTERPRETATION
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    edge_raw = oa_acc * 100 - baseline_acc
    gate_edge = (
        gated_oa_prod["combined_precision"] - max(baseline_rate, 1 - baseline_rate)
        if gated_oa_prod["n_trades"] > 0 else 0.0
    )

    if oa_auc > 0.55 and gate_edge > 0.03:
        verdict = (
            f"  AUC {oa_auc:.3f} > 0.55 and gated precision edge {gate_edge:+.3f}.\n"
            f"  Model shows a MODEST but REAL edge on confident signals."
        )
    elif oa_auc > 0.52:
        verdict = (
            f"  AUC {oa_auc:.3f} slightly above 0.50; gated edge {gate_edge:+.3f}.\n"
            f"  Weak signal — transaction costs likely erase any real-world edge."
        )
    else:
        verdict = (
            f"  AUC {oa_auc:.3f} near 0.50; gated edge {gate_edge:+.3f}.\n"
            f"  No reliable out-of-sample predictive power detected."
        )

    print(f"\n  Raw accuracy edge  : {edge_raw:+.2f}pp  (vs {baseline_acc:.2f}% random)")
    print(f"  MCC (overall)      : {oa_mcc:.4f}  (0 = random, 1 = perfect)")
    print(f"\n{verdict}")
    print(f"\n  Walk-forward accuracy : {oa_acc*100:.2f}%")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5: SAVE
    # ─────────────────────────────────────────────────────────────────────────────
    results = {
        "method":           "walk_forward_validation_v2",
        "n_folds":          N_FOLDS,
        "horizon_bars":     HORIZON,
        "purge_gap_bars":   HORIZON,
        "val_ratio":        VAL_RATIO,
        "confidence_gate":  CONFIDENCE_THRESHOLD,
        "n_features":       int(X_all.shape[1]),
        "fold_results":     fold_results,
        "aggregate": {
            "mean_accuracy":      mean_acc,
            "std_accuracy":       std_acc,
            "overall_accuracy":   float(oa_acc),
            "overall_auc":        float(oa_auc),
            "overall_f1":         float(oa_f1),
            "overall_mcc":        float(oa_mcc),
            "overall_precision":  float(oa_prec),
            "overall_recall":     float(oa_rec),
            "random_baseline_acc": baseline_acc / 100,
            "random_baseline_prec": max(baseline_rate, 1 - baseline_rate),
            "gated_production":   gated_oa_prod,
            "gated_calibrated":   gated_oa_cal,
            "mean_cal_threshold": mean_cal_thr,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to: {RESULTS_PATH}")
    print("\n" + "=" * 90)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
