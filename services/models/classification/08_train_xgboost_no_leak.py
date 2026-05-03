from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
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
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [4, 8, 16]

# =========================================================
# HELPERS
# =========================================================
def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    return metrics

def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list, horizon: int):
    importance = model.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 20 Features - XGBoost (H{horizon})')
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f'feature_importance_h{horizon}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {plot_path}")

def train_and_eval_horizon(horizon: int):
    print(f"\n" + "-"*40)
    print(f"TRAINING FOR HORIZON: H{horizon}")
    print("-"*40)
    
    # Load horizon-specific files
    X_train = np.load(DATA_DIR / f"X_train_h{horizon}.npy")
    y_train = np.load(DATA_DIR / f"y_train_h{horizon}.npy")
    X_val = np.load(DATA_DIR / f"X_val_h{horizon}.npy")
    y_val = np.load(DATA_DIR / f"y_val_h{horizon}.npy")
    X_test = np.load(DATA_DIR / f"X_test_h{horizon}.npy")
    y_test = np.load(DATA_DIR / f"y_test_h{horizon}.npy")

    # Load feature names
    feature_cols_path = DATA_DIR / "feature_columns.json"
    with open(feature_cols_path, "r") as f:
        feature_names = json.load(f)

    # Calculate scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg) / max(1, n_pos)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    print(f"Training XGBoost model (Train={len(X_train)} samples, Val={len(X_val)} samples, Features={len(feature_names)})...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20
    )

    print("\nEvaluating on Test Set...")
    test_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, test_prob)
    
    print("\nPlotting Feature Importance...")
    plot_feature_importance(model, feature_names, horizon)
    
    model.save_model(MODEL_DIR / f"xgboost_h{horizon}.json")
    
    return metrics

def main():
    results = []
    
    for h in HORIZONS:
        try:
            m = train_and_eval_horizon(h)
            results.append({
                "horizon": f"H{h}",
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "auc": m["auc"],
                "mcc": m["mcc"]
            })
        except FileNotFoundError:
            print(f"Warning: Data for H{h} not found. Ensure 07_prepare scripts have run successfully.")

    if not results:
        print("No results to display.")
        return

    # Print Comparison Table
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("TARGET HORIZON COMPARISON TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)

    # Save summary metrics
    summary_path = MODEL_DIR / "horizon_comparison_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary metrics saved to: {summary_path}")

if __name__ == "__main__":
    main()