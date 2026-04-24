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
DATA_DIR = PROJECT_ROOT / "data" / "processed"
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