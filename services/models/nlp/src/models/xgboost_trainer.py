import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def create_target(df):
    """Create target variable: 1 if next 15m close is higher, else 0."""
    price_col = 'close' if 'close' in df.columns else ('close_price' if 'close_price' in df.columns else None)
    if not price_col:
        raise ValueError("Could not find a valid price column in data.")

    df = df.sort_values("timestamp")

    df['target'] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    df['target_h1'] = (df[price_col].shift(-4) > df[price_col]).astype(int)

    return df.dropna(subset=['target', 'target_h1']).copy()


def _build_model(y_train: pd.Series) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
    )


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    description: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Walk-forward (expanding-window) cross-validation.
    Starts with 50% of data as the minimum training window, then expands
    step-by-step so each fold tests only on unseen future data.
    """
    min_train = max(int(len(X) * 0.5), 30)
    remaining = len(X) - min_train
    if remaining <= 0:
        print(f"  {description}: insufficient data for walk-forward validation.")
        return pd.DataFrame()

    step = max(remaining // n_splits, 1)
    fold_metrics = []

    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, len(X))

        if train_end >= len(X) or train_end >= test_end:
            break

        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

        if len(y_test) == 0 or y_test.nunique() < 2:
            continue

        model = _build_model(y_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        fold_metrics.append({
            "fold": i + 1,
            "train_size": train_end,
            "test_size": len(y_test),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, probs),
        })

    if not fold_metrics:
        print(f"  {description}: no valid folds produced.")
        return pd.DataFrame()

    df_m = pd.DataFrame(fold_metrics)
    print(f"\n=== {description} — Walk-Forward Validation ({len(df_m)} folds) ===")
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"  {col.capitalize():<10}: mean={df_m[col].mean():.4f}  std={df_m[col].std():.4f}")
    print("=" * 60)
    return df_m


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, description: str):
    """
    Train XGBoost with walk-forward validation, then do a final fit on 80% for
    feature importance and confusion matrix reporting.
    """
    # Walk-forward validation (no look-ahead bias)
    walk_forward_validate(X, y, description)

    # Final held-out evaluation for feature importance
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = _build_model(y_train)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.0

    print(f"\n=== {description} — Final Split (80/20) ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    feat_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("\nTop features:")
    print(feat_importance.head(10))
    sent_imp = feat_importance[feat_importance.index.str.contains("sentiment")]
    print("\nSentiment feature importance:")
    print(sent_imp.head(10))
    print("=" * 60 + "\n")


def main():
    data_path = "data/processed/sol_usdt_15m_with_sentiment.csv"

    print("Loading merged dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    df = create_target(df)

    print("\nTarget distribution (15m):")
    print(df["target"].value_counts(normalize=True))

    all_cols = df.columns.tolist()
    metadata_cols = ["timestamp", "target", "target_h1"]

    sentiment_features = [
        "sentiment_mean", "sentiment_std", "positive_ratio", "negative_ratio",
        "neutral_ratio", "news_volume", "max_confidence", "gdelt_volume",
        "rss_volume", "gdelt_sentiment", "rss_sentiment", "sentiment_spike",
    ]
    temporal_sentiment = [c for c in all_cols if "sentiment" in c]
    sentiment_features.extend(temporal_sentiment)
    sentiment_cols_present = list(set(c for c in sentiment_features if c in all_cols))

    text_cols = ["title", "url", "cleaned_text", "source", "feed_source", "sentiment_label"]
    leakage_cols = ["future_return", "target", "target_up", "target_h1"]
    leakage_patterns = ["future", "t+1", "lead", "target"]

    def is_leakage(col):
        return col in leakage_cols or any(p in col.lower() for p in leakage_patterns)

    tech_features = [
        c for c in all_cols
        if c not in metadata_cols
        and c not in sentiment_cols_present
        and c not in text_cols
        and not is_leakage(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(df) < 10:
        print("\nWarning: Dataset is very small. Metrics will be unreliable.")

    for target_col in ["target", "target_h1"]:
        print(f"\n{'#' * 47}")
        print(f"       TRAINING FOR PREDICTION: {target_col.upper()}")
        print(f"{'#' * 47}")

        y = df[target_col]

        print("\nTop correlations with target:")
        if tech_features:
            corr = df[tech_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            print(corr.head(10))
        else:
            print("No technical features to correlate.")

        # Naive baseline
        split_idx = int(len(df) * 0.8)
        y_train_base = y.iloc[:split_idx]
        y_test_base = y.iloc[split_idx:]
        if not y_train_base.empty:
            majority_class = y_train_base.mode()[0]
            baseline_pred = [majority_class] * len(y_test_base)
            print("\n=== Baseline (Majority Class) ===")
            print(f"Accuracy:  {accuracy_score(y_test_base, baseline_pred):.4f}")
            print(f"F1 Score:  {f1_score(y_test_base, baseline_pred, zero_division=0):.4f}")
            print("=" * 60)

        X_with = df[tech_features + sentiment_cols_present]
        train_and_evaluate(X_with, y, "Model: WITH Sentiment Features")

        if tech_features:
            X_without = df[tech_features]
            train_and_evaluate(X_without, y, "Model: Technical Features ONLY")
        else:
            print("No technical features available for secondary model.")


if __name__ == "__main__":
    main()
