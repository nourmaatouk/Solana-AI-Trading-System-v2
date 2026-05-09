"""
08_train_xgboost_no_leak.py
Train XGBoost classifiers for H4 / H8 / H16 horizons.

Prerequisites: run 07_prepare_no_leak_data.py first to generate the .npy splits.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from config import (
    HORIZONS,
    MODEL_DIR,
    PROCESSED_DIR,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SEED,
    XGB_SUBSAMPLE,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":  auc,
        "mcc":      float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
    }


def calibrate_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Sweep thresholds on the val set and return the one maximising F1."""
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = np.array([
        f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ])
    best_thresh = float(thresholds[f1s.argmax()])
    print(f"  Val-F1 calibration: best threshold = {best_thresh:.3f} "
          f"(F1 = {f1s.max():.4f})")
    return best_thresh


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list[str], horizon: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from config import PLOTS_DIR
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-20:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_idx)), importance[top_idx], align="center")
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"Top 20 Features — XGBoost H{horizon}")
    plt.tight_layout()
    path = PLOTS_DIR / f"feature_importance_h{horizon}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Feature importance plot saved: {path}")


# =========================================================
# TRAINING
# =========================================================
def train_horizon(horizon: int) -> dict:
    print(f"\n{'─'*40}\nTRAINING H{horizon}\n{'─'*40}")

    X_train = np.load(PROCESSED_DIR / f"X_train_h{horizon}.npy")
    y_train = np.load(PROCESSED_DIR / f"y_train_h{horizon}.npy")
    X_val   = np.load(PROCESSED_DIR / f"X_val_h{horizon}.npy")
    y_val   = np.load(PROCESSED_DIR / f"y_val_h{horizon}.npy")
    X_test  = np.load(PROCESSED_DIR / f"X_test_h{horizon}.npy")
    y_test  = np.load(PROCESSED_DIR / f"y_test_h{horizon}.npy")

    with open(PROCESSED_DIR / "feature_columns.json") as f:
        feature_names: list[str] = json.load(f)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(1, n_pos)
    print(f"  Train: {len(X_train)} rows | pos={n_pos}, neg={n_neg}, "
          f"scale_pos_weight={scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators           = XGB_N_ESTIMATORS,
        max_depth              = XGB_MAX_DEPTH,
        learning_rate          = XGB_LEARNING_RATE,
        subsample              = XGB_SUBSAMPLE,
        colsample_bytree       = XGB_COLSAMPLE_BYTREE,
        objective              = "binary:logistic",
        eval_metric            = "logloss",
        random_state           = XGB_SEED,
        tree_method            = "hist",
        scale_pos_weight       = scale_pos_weight,
        n_jobs                 = -1,
        early_stopping_rounds  = XGB_EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20,
    )

    # Val-set threshold calibration
    val_probs  = model.predict_proba(X_val)[:, 1]
    best_thresh = calibrate_threshold(y_val, val_probs)

    # Save calibrated threshold (informational; predictor gate uses CONFIDENCE_THRESHOLD)
    thresh_data = {"horizon": horizon, "val_f1_threshold": best_thresh}
    with open(PROCESSED_DIR / f"threshold_h{horizon}.json", "w") as f:
        json.dump(thresh_data, f, indent=2)

    # Evaluate on test set at default 0.5 (and at calibrated threshold for reference)
    test_probs = model.predict_proba(X_test)[:, 1]
    print(f"\nTest set evaluation (threshold=0.50):")
    metrics_50 = evaluate(y_test, test_probs, threshold=0.50)
    print(f"\nTest set evaluation (calibrated threshold={best_thresh:.3f}):")
    metrics_cal = evaluate(y_test, test_probs, threshold=best_thresh)

    plot_feature_importance(model, feature_names, horizon)

    # Save model
    model_path = MODEL_DIR / f"xgboost_h{horizon}.json"
    model.save_model(str(model_path))
    print(f"\n  Model saved: {model_path}")

    return {
        "horizon":       f"H{horizon}",
        "accuracy":      metrics_50["accuracy"],
        "f1":            metrics_50["f1"],
        "auc_roc":       metrics_50["auc_roc"],
        "mcc":           metrics_50["mcc"],
        "cal_threshold": best_thresh,
        "cal_f1":        metrics_cal["f1"],
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    results = []
    for h in HORIZONS:
        try:
            results.append(train_horizon(h))
        except FileNotFoundError as e:
            print(f"Warning: data for H{h} not found ({e}). "
                  f"Run 07_prepare_no_leak_data.py first.")

    if not results:
        print("No results — nothing was trained.")
        return

    df = pd.DataFrame(results)
    print(f"\n{'='*60}\nHORIZON COMPARISON\n{'='*60}")
    print(df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
