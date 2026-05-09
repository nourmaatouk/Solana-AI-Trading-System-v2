"""
20_walk_forward_validation.py

WALK-FORWARD TEST - TRUE OUT-OF-SAMPLE VERIFICATION

This script implements genuine forward testing where:
1. Model NEVER sees future data (strict temporal boundary)
2. Each fold trains on PAST, tests on FUTURE (unseen)
3. Realistic estimation of real-world performance
4. Shows if 82% accuracy is real or overfitted

This is the gold standard for time series validation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config import (
    LEAKAGE_SUBSTRINGS,
    MODEL_DIR,
    NON_STATIONARY_COLUMNS,
    PROCESSED_DIR,
    TECH_FEATURES_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
    XGB_SEED,
)

INPUT_PATH   = PROCESSED_DIR / TECH_FEATURES_CSV
RESULTS_PATH = MODEL_DIR / "walk_forward_results.json"

N_FOLDS = 5   # number of walk-forward periods (validation parameter, not a model constant)

print("=" * 90)
print("🔄 WALK-FORWARD VALIDATION - TRUE OUT-OF-SAMPLE TEST")
print("=" * 90)
print(f"\nThis test prevents look-ahead bias by:")
print(f"  1. Dividing data into {N_FOLDS} time periods")
print(f"  2. Training on EVERY past period")
print(f"  3. Testing on ONE future period (never seen before)")
print(f"  4. Averaging results across all folds")
print()


# =========================================================
# STEP 1: LOAD DATA
# =========================================================
print("[STEP 1] Loading data...")

df = pd.read_csv(INPUT_PATH)

# Mirror 07_prepare_no_leak_data.py: drop non-stationary + leakage + meta columns
_ALWAYS_DROP = {"timestamp", "datetime", "target_direction", "future_return_h"}
feature_columns = [
    col for col in df.columns
    if col not in _ALWAYS_DROP
    and col.lower() not in NON_STATIONARY_COLUMNS
    and not any(sub in col.lower() for sub in LEAKAGE_SUBSTRINGS)
]

X_df = df[feature_columns].copy()
close_prices = df["close"].copy()

# Clean
X_df = X_df.replace([np.inf, -np.inf], np.nan)
valid_mask = X_df.notna().all(axis=1)
X_df = X_df.loc[valid_mask].reset_index(drop=True)
close_prices = close_prices.loc[valid_mask].reset_index(drop=True)

# Forward-looking target: 1 if price 4 bars ahead is higher than current price
target_honest = (close_prices.shift(-4) > close_prices).astype(int)
valid_target_mask = target_honest.notna()  # last 4 rows have NaN (no future)

X_honest = X_df.loc[valid_target_mask].reset_index(drop=True)
target_honest = target_honest.loc[valid_target_mask].reset_index(drop=True)

print(f"Data loaded: X={X_honest.shape}, y={target_honest.shape}")
print(f"Target distribution: 0: {(target_honest==0).sum()}, 1: {(target_honest==1).sum()}")


# =========================================================
# STEP 2: BUILD SEQUENCES
# =========================================================
print("\n[STEP 2] Building sequences...")

X_array = X_honest.values.astype(np.float64)

X_seq = []
y_seq = []
for i in range(SEQUENCE_LENGTH, len(X_array)):
    X_seq.append(X_array[i - SEQUENCE_LENGTH:i])
    y_seq.append(target_honest.iloc[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.int64)

print(f"Sequences: X={X_seq.shape}, y={y_seq.shape}")


# =========================================================
# STEP 3: WALK-FORWARD VALIDATION
# =========================================================
print(f"\n[STEP 3] Running {N_FOLDS}-fold walk-forward validation...")
print("=" * 90)

fold_results = []
accumulated_preds = []
accumulated_true = []

# Calculate fold boundaries
n_samples = len(X_seq)
fold_size = n_samples // (N_FOLDS + 1)  # Reserve last fold for test

print(f"Total samples: {n_samples}")
print(f"Fold size: {fold_size}")
print()

for fold in range(N_FOLDS):
    print(f"{'─' * 90}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'─' * 90}")
    
    # Define boundaries
    train_end = fold_size * (fold + 1)
    test_start = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2)
    
    # Split data
    X_train_fold = X_seq[:train_end].reshape(train_end, -1)
    y_train_fold = y_seq[:train_end]
    
    X_test_fold = X_seq[test_start:test_end].reshape(test_end - test_start, -1)
    y_test_fold = y_seq[test_start:test_end]
    
    print(f"Training samples:   {X_train_fold.shape[0]}")
    print(f"Testing samples:    {X_test_fold.shape[0]}")
    print(f"Training period:    Sample 0 to {train_end}")
    print(f"Testing period:     Sample {test_start} to {test_end} (FUTURE - NEVER SEEN)")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=XGB_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    
    model.fit(X_train_fold, y_train_fold, verbose=0)
    
    # Test model
    y_test_proba = model.predict_proba(X_test_fold)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test_fold, y_test_pred)
    prec = precision_score(y_test_fold, y_test_pred, zero_division=0)
    recall = recall_score(y_test_fold, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_fold, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test_fold, y_test_proba)
    
    fold_result = {
        "fold": fold + 1,
        "train_size": int(X_train_fold.shape[0]),
        "test_size": int(X_test_fold.shape[0]),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }
    
    fold_results.append(fold_result)
    accumulated_preds.extend(y_test_pred)
    accumulated_true.extend(y_test_fold)
    
    print(f"\n  Accuracy:  {acc*100:6.2f}%")
    print(f"  Precision: {prec*100:6.2f}%")
    print(f"  Recall:    {recall*100:6.2f}%")
    print(f"  F1:        {f1*100:6.2f}%")
    print(f"  AUC:       {auc*100:6.2f}%")


# =========================================================
# STEP 4: AGGREGATE RESULTS
# =========================================================
print("\n" + "=" * 90)
print("WALK-FORWARD RESULTS - AGGREGATE")
print("=" * 90)

accumulated_true = np.array(accumulated_true)
accumulated_preds = np.array(accumulated_preds)

overall_acc = accuracy_score(accumulated_true, accumulated_preds)
overall_prec = precision_score(accumulated_true, accumulated_preds, zero_division=0)
overall_recall = recall_score(accumulated_true, accumulated_preds, zero_division=0)
overall_f1 = f1_score(accumulated_true, accumulated_preds, zero_division=0)

accuracies = [f["accuracy"] for f in fold_results]
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\n📊 PER-FOLD ACCURACY:")
for f in fold_results:
    print(f"  Fold {f['fold']}: {f['accuracy']*100:6.2f}%")

print(f"\n📊 AGGREGATE METRICS (across all folds):")
print(f"  Mean Accuracy:     {mean_acc*100:6.2f}% (±{std_acc*100:.2f}%)")
print(f"  Overall Accuracy:  {overall_acc*100:6.2f}%")
print(f"  Overall Precision: {overall_prec*100:6.2f}%")
print(f"  Overall Recall:    {overall_recall*100:6.2f}%")
print(f"  Overall F1:        {overall_f1*100:6.2f}%")

baseline = max(np.unique(accumulated_true, return_counts=True)[1]) / len(accumulated_true) * 100
edge = overall_acc*100 - baseline

print(f"\n🎲 RANDOM BASELINE: {baseline:.2f}%")
print(f"   Your edge: {edge:.2f}%")


# =========================================================
# STEP 5: COMPARISON
# =========================================================
print("\n" + "=" * 90)
print("COMPARISON: STANDARD TEST vs WALK-FORWARD TEST")
print("=" * 90)

print(f"\n❌ STANDARD TEST (70-15-15 split):")
print(f"   Test Accuracy: 64.21%  (H4 production model, 46 features, no-leak)")
print(f"   Problem: Model sees similar market regime in train & test")
print(f"   Problem: Can memorize patterns")

print(f"\n✓ WALK-FORWARD TEST (true out-of-sample):")
print(f"   Mean Accuracy: {mean_acc*100:.2f}%")
print(f"   Accuracy drop: {82.28 - mean_acc*100:.2f}%")
print(f"   Benefit: Tests on FUTURE data never seen during training")
print(f"   Benefit: More realistic estimate of live trading performance")


# =========================================================
# STEP 6: INTERPRETATION
# =========================================================
print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)

if mean_acc*100 > 75:
    print(f"\n🚨 ACCURACY STILL {mean_acc*100:.1f}% IN WALK-FORWARD")
    print(f"   → Model has GENUINE PREDICTIVE POWER")
    print(f"   → NOT just memorizing the training data")
    print(f"   → Shows consistent edge across different time periods")
    print(f"   → Worth investigating for live trading")
    
elif mean_acc*100 > 60:
    print(f"\n✓ ACCURACY {mean_acc*100:.1f}% IN WALK-FORWARD")
    print(f"   → Model has MODEST PREDICTIVE POWER")
    print(f"   → {edge:.1f}% edge over random baseline")
    print(f"   → Reasonable for financial prediction")
    print(f"   → But watch for transaction costs eating profits")
    
elif mean_acc*100 > 52:
    print(f"\n⚠️  ACCURACY DROPPED TO {mean_acc*100:.1f}% IN WALK-FORWARD")
    print(f"   → Only {edge:.1f}% edge over random")
    print(f"   → Model OVERFITTED to original train-test split")
    print(f"   → Standard test was too optimistic")
    print(f"   → NOT reliable for live trading")
    
else:
    print(f"\n❌ ACCURACY {mean_acc*100:.1f}% IN WALK-FORWARD")
    print(f"   → Model has NO predictive power")
    print(f"   → Performance below random baseline")
    print(f"   → DO NOT USE for trading")


# =========================================================
# STEP 7: SAVE RESULTS
# =========================================================
print("\n[STEP 7] Saving results...")

results = {
    "method": "walk_forward_validation",
    "n_folds": N_FOLDS,
    "fold_results": fold_results,
    "aggregate": {
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "overall_accuracy": float(overall_acc),
        "overall_precision": float(overall_prec),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "random_baseline": float(baseline),
        "edge_over_random": float(edge),
    },
    "comparison": {
        "standard_test_accuracy": 0.6421,
        "walk_forward_accuracy": float(mean_acc),
        "accuracy_drop_percent": 64.21 - mean_acc*100,
        "interpretation": "H4 production model (46 features, no-leak); walk-forward uses same feature set"
    }
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved to: {RESULTS_PATH}")

print("\n" + "=" * 90)
print("✓ WALK-FORWARD VALIDATION COMPLETE")
print("=" * 90)