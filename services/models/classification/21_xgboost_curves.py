"""
xgboost_curves.py

Comprehensive XGBoost Binary Classification Pipeline:
- Loads pre-split train/val/test data from .npy files
- Trains XGBoost with evaluation tracking on train/val sets
- Generates training vs validation curves (logloss, error, auc)
- Evaluates on test set with comprehensive metrics
- Creates all important performance visualizations
- Saves everything to organized folders with JSON summary report
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    ACTIVE_HORIZON as HORIZON,
    PLOTS_DIR,
    PROCESSED_DIR as DATA_DIR,
    REPORTS_DIR,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_SEED,
    XGB_SUBSAMPLE,
)

VALIDATION_PLOTS_DIR = PLOTS_DIR / "validation_curves"
TEST_PLOTS_DIR       = PLOTS_DIR / "test_curves"

X_TRAIN_PATH = DATA_DIR / f"X_train_h{HORIZON}.npy"
Y_TRAIN_PATH = DATA_DIR / f"y_train_h{HORIZON}.npy"
X_VAL_PATH   = DATA_DIR / f"X_val_h{HORIZON}.npy"
Y_VAL_PATH   = DATA_DIR / f"y_val_h{HORIZON}.npy"
X_TEST_PATH  = DATA_DIR / f"X_test_h{HORIZON}.npy"
Y_TEST_PATH  = DATA_DIR / f"y_test_h{HORIZON}.npy"

REPORT_PATH  = REPORTS_DIR / "xgboost_curves_run_summary.json"


def main() -> None:
    for d in [VALIDATION_PLOTS_DIR, TEST_PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # PART 0: LOAD DATA
    # =========================================================
    print("=" * 80)
    print("XGBOOST BINARY CLASSIFICATION PIPELINE")
    print("=" * 80)
    print("\n[PART 0] Loading data from .npy files...")

    try:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val   = np.load(X_VAL_PATH)
        y_val   = np.load(Y_VAL_PATH)
        X_test  = np.load(X_TEST_PATH)
        y_test  = np.load(Y_TEST_PATH)
        print("✓ All data files loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error loading data files: {e}")
        print(f"  Expected directory: {DATA_DIR}")
        print(f"  Run 07_prepare_no_leak_data.py to generate them.")
        raise SystemExit(1)

    print(f"  X_train shape: {X_train.shape}  y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}  y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}  y_test shape:  {y_test.shape}")

    # =========================================================
    # PART 0B: FLATTEN DATA IF NEEDED
    # =========================================================
    print("\n[PART 0B] Checking and flattening data if needed...")

    if len(X_train.shape) > 2:
        print(f"  Data is {len(X_train.shape)}D (sequences), flattening to 2D...")
        original_shape = X_train.shape
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val   = X_val.reshape(X_val.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)
        print(f"  ✓ Flattened from {original_shape} to {X_train.shape}")
    else:
        print("  ✓ Data is already 2D, no flattening needed")

    # =========================================================
    # PART 1: TRAIN XGBOOST WITH NATIVE API (xgb.train)
    # =========================================================
    print("\n" + "=" * 80)
    print("PART 1: TRAINING XGBOOST WITH NATIVE API (xgb.train)")
    print("=" * 80)

    print("\nPreparing DMatrix objects for xgb.train...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective":       "binary:logistic",
        "max_depth":       XGB_MAX_DEPTH,
        "eta":             XGB_LEARNING_RATE,
        "subsample":       XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "tree_method":     "hist",
        "eval_metric":     ["logloss", "error", "auc"],
        "seed":            XGB_SEED,
    }

    print("Training model with evaluation sets using native xgb.train...")
    evals_result: dict = {}
    num_round = 300
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=50,
    )

    best_iteration   = bst.best_iteration
    train_metrics    = evals_result.get("train", {})
    val_metrics      = evals_result.get("validation", {})

    print(f"\n✓ Training complete with eval monitoring")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Available metrics: {list(val_metrics.keys())}")

    # =========================================================
    # PART 2: GENERATE AND SAVE TRAINING/VALIDATION CURVES
    # =========================================================
    print("\n" + "=" * 80)
    print("PART 2: GENERATING TRAINING/VALIDATION CURVES (selected)")
    print("=" * 80)

    print("\n[2.1] Creating Log Loss comparison plot...")
    if "logloss" in train_metrics and "logloss" in val_metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = range(len(train_metrics["logloss"]))
        ax.plot(epochs, train_metrics["logloss"], label="Training Log Loss",   linewidth=2.5, color="steelblue")
        ax.plot(epochs, val_metrics["logloss"],   label="Validation Log Loss", linewidth=2.5, color="darkred")
        if best_iteration is not None:
            ax.axvline(x=best_iteration, color="green", linestyle="--", linewidth=2.5,
                       label=f"Best Iteration ({best_iteration})")
        ax.set_xlabel("Boosting Round", fontsize=12, fontweight="bold")
        ax.set_ylabel("Log Loss",       fontsize=12, fontweight="bold")
        ax.set_title("Training vs Validation Log Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(VALIDATION_PLOTS_DIR / "01_train_val_logloss.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: 01_train_val_logloss.png")
    else:
        print(f"  ⚠ Log Loss metrics not available")

    # =========================================================
    # PART 3: EVALUATE ON TEST SET
    # =========================================================
    print("\n" + "=" * 80)
    print("PART 3: EVALUATING ON TEST SET")
    print("=" * 80)

    y_test_proba = bst.predict(dtest)
    y_test_pred  = (y_test_proba >= 0.5).astype(int)

    test_accuracy  = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall    = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1        = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc       = roc_auc_score(y_test, y_test_proba)
    test_cm        = confusion_matrix(y_test, y_test_pred)

    print(f"\n✓ Test Set Performance Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"  Confusion Matrix:\n    {test_cm}")

    # =========================================================
    # PART 4: GENERATE AND SAVE ALL TEST PERFORMANCE CURVES
    # =========================================================
    print("\n" + "=" * 80)
    print("PART 4: GENERATING TEST PERFORMANCE CURVES (selected)")
    print("=" * 80)

    # 4.1: ROC Curve
    print("\n[4.1] Creating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve - Test Set Performance", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(TEST_PLOTS_DIR / "01_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 01_roc_curve.png")

    # 4.2: Precision-Recall Curve
    print("[4.2] Creating Precision-Recall Curve...")
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall_curve, precision_curve)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall_curve, precision_curve, color="blue", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    baseline_pr = (y_test == 1).sum() / len(y_test)
    ax.axhline(y=baseline_pr, color="red", linestyle="--", lw=2, label=f"Baseline Precision ({baseline_pr:.4f})")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall",    fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curve - Test Set Performance", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(TEST_PLOTS_DIR / "02_precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 02_precision_recall_curve.png")

    # 4.3: Confusion Matrix
    print("[4.3] Creating Confusion Matrix Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(test_cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11, fontweight="bold")
    ax.set(xticks=np.arange(test_cm.shape[1]),
           yticks=np.arange(test_cm.shape[0]),
           ylabel="True Label", xlabel="Predicted Label")
    ax.set_xticklabels(["Down (0)", "Up (1)"], fontsize=11)
    ax.set_yticklabels(["Down (0)", "Up (1)"], fontsize=11)
    ax.set_title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
    thresh = test_cm.max() / 2.0
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            ax.text(j, i, format(test_cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if test_cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(TEST_PLOTS_DIR / "03_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 03_confusion_matrix.png")

    # 4.6: Feature Importance (Top 20)
    print("[4.6] Creating Feature Importance Plot...")
    importance_dict     = bst.get_score(importance_type="weight")
    feature_importance  = np.zeros(X_test.shape[1])
    for fid, score in importance_dict.items():
        idx = int(fid.replace("f", ""))
        if idx < len(feature_importance):
            feature_importance[idx] = score
    sorted_indices  = np.argsort(feature_importance)[::-1]
    top_20_indices  = sorted_indices[:20]
    top_20_vals     = feature_importance[top_20_indices]
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_vals)))
    ax.barh(range(len(top_20_vals)), top_20_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top_20_vals)))
    ax.set_yticklabels([f"Feature {i}" for i in top_20_indices], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title("Top 20 Feature Importance - XGBoost Model", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(TEST_PLOTS_DIR / "06_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 06_feature_importance.png")

    # =========================================================
    # PART 6: SAVE RESULTS AND GENERATE JSON REPORT
    # =========================================================
    print("\n" + "=" * 80)
    print("PART 6: SAVING RESULTS AND GENERATING JSON REPORT")
    print("=" * 80)

    summary_report = {
        "model_info": {
            "model_type":        "XGBoost Binary Classifier (Native API)",
            "n_estimators":       num_round,
            "max_depth":          params["max_depth"],
            "learning_rate":      params["eta"],
            "subsample":          params["subsample"],
            "colsample_bytree":   params["colsample_bytree"],
            "objective":          params["objective"],
        },
        "training_info": {
            "best_iteration":       int(best_iteration) if best_iteration is not None else None,
            "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
            "total_rounds_trained":  num_round,
        },
        "dataset_info": {
            "n_train_samples": int(X_train.shape[0]),
            "n_val_samples":   int(X_val.shape[0]),
            "n_test_samples":  int(X_test.shape[0]),
            "n_features":      int(X_train.shape[1]),
            "target_class_distribution_test": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum()),
            },
        },
        "test_metrics": {
            "accuracy":   float(test_accuracy),
            "precision":  float(test_precision),
            "recall":     float(test_recall),
            "f1_score":   float(test_f1),
            "auc_roc":    float(test_auc),
        },
        "confusion_matrix": {
            "true_negatives":  int(test_cm[0, 0]),
            "false_positives": int(test_cm[0, 1]),
            "false_negatives": int(test_cm[1, 0]),
            "true_positives":  int(test_cm[1, 1]),
            "matrix":          test_cm.tolist(),
        },
        "validation_metrics_at_best_iteration": {
            "best_val_logloss": float(val_metrics["logloss"][best_iteration])
                                if (best_iteration is not None and "logloss" in val_metrics) else None,
            "best_val_error":   float(val_metrics["error"][best_iteration])
                                if (best_iteration is not None and "error" in val_metrics) else None,
            "best_val_auc":     float(val_metrics["auc"][best_iteration])
                                if (best_iteration is not None and "auc" in val_metrics) else None,
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n✓ Summary report generated: {REPORT_PATH}")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - ALL RESULTS SAVED SUCCESSFULLY")
    print("=" * 80)

    print(f"\n📈 Key Test Metrics:")
    print(f"  ├─ Best Training Iteration: {best_iteration}")
    print(f"  ├─ Test Accuracy:           {test_accuracy:.4f}")
    print(f"  ├─ Test Precision:          {test_precision:.4f}")
    print(f"  ├─ Test Recall:             {test_recall:.4f}")
    print(f"  ├─ Test F1-Score:           {test_f1:.4f}")
    print(f"  └─ Test AUC-ROC:            {test_auc:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
