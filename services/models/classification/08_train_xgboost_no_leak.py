from pathlib import Path
import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train_no_leak.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train_no_leak.npy"
X_VAL_PATH = DATA_DIR / "X_val_no_leak.npy"
Y_VAL_PATH = DATA_DIR / "y_val_no_leak.npy"
X_TEST_PATH = DATA_DIR / "X_test_no_leak.npy"
Y_TEST_PATH = DATA_DIR / "y_test_no_leak.npy"

MODEL_PATH = MODEL_DIR / "xgboost_direction_model_no_leak.json"
METRICS_PATH = MODEL_DIR / "xgboost_metrics_no_leak.json"


# =========================================================
# HELPERS
# =========================================================
def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


# =========================================================
# LOAD
# =========================================================
print("Loading no-leak data...")
print(f"DATA_DIR resolving to: {DATA_DIR}")

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("Original shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# =========================================================
# FLATTEN
# =========================================================
X_train_flat = flatten_sequences(X_train)
X_val_flat = flatten_sequences(X_val)
X_test_flat = flatten_sequences(X_test)

print("\nFlattened shapes:")
print("X_train_flat:", X_train_flat.shape)
print("X_val_flat  :", X_val_flat.shape)
print("X_test_flat :", X_test_flat.shape)


# =========================================================
# MODEL
# =========================================================
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)

print("\nTraining XGBoost...")
model.fit(
    X_train_flat,
    y_train,
    eval_set=[(X_val_flat, y_val)],
    verbose=True,
)


# =========================================================
# EVALUATE
# =========================================================
val_prob = model.predict_proba(X_val_flat)[:, 1]
test_prob = model.predict_proba(X_test_flat)[:, 1]

val_metrics = evaluate_model(y_val, val_prob)
test_metrics = evaluate_model(y_test, test_prob)

print("\nValidation metrics:")
for k, v in val_metrics.items():
    print(f"{k}: {v}")

print("\nTest metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v}")

test_pred = (test_prob >= 0.5).astype(int)
print("\nClassification report on test:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))


# =========================================================
# SAVE
# =========================================================
model.save_model(MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

print("\nSaved model to:")
print(MODEL_PATH)

print("\nSaved metrics to:")
print(METRICS_PATH)


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================
print("\n" + "="*60)
print("RUNNING SHAP EXPLAINABILITY ANALYSIS")
print("="*60)

try:
    import shap
    import matplotlib.pyplot as plt

    # 1. Load feature names to map the flattened indices
    FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
    if not FEATURE_COLS_PATH.exists():
        print(f"Warning: feature_columns.json not found at {FEATURE_COLS_PATH}. Using generic names.")
        feature_names = [f"f{i}" for i in range(X_train.shape[2])]
    else:
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_names = json.load(f)

    SEQ_LEN = X_train.shape[1] 
    NUM_FEATS = X_train.shape[2]

    # Generate names for flattened features, e.g., "RSI_t-31", "RSI_t-30", ...
    # Following the order: time_step_0 (t-31) features, then time_step_1 (t-30) features...
    flattened_names = []
    for t in range(SEQ_LEN):
        t_offset = (SEQ_LEN - 1) - t
        suffix = f"t-{t_offset}" if t_offset > 0 else "t-0"
        for name in feature_names:
            flattened_names.append(f"{name}_{suffix}")

    # 2. Compute SHAP values
    # Use a subset of test data for speed (e.g., 200 samples)
    explainer = shap.Explainer(model)
    subset_size = min(200, len(X_test_flat))
    X_subset = X_test_flat[:subset_size]
    
    print(f"Computing SHAP values for {subset_size} samples...")
    shap_values = explainer(X_subset)

    # 3. Plotting
    REPORT_DIR = PROJECT_ROOT / "classification" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_subset, feature_names=flattened_names, show=False, max_display=20)
    plt.title("XGBoost SHAP Summary (Top 20 Time-Series Features)")
    
    summary_plot_path = REPORT_DIR / "xgboost_shap_summary.png"
    plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"SHAP Summary Plot saved to: {summary_plot_path}")

    # Bar Plot (Global Importance)
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("XGBoost Global Feature Importance (SHAP Bar Plot)")
    
    bar_plot_path = REPORT_DIR / "xgboost_shap_bar.png"
    plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"SHAP Bar Plot saved to: {bar_plot_path}")

except ImportError:
    print("Error: 'shap' or 'matplotlib' not installed. Skipping explainability.")
except Exception as e:
    print(f"Error during SHAP analysis: {e}")

print("\nPipeline execution finished.")