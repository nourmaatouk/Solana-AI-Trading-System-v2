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
    
    # Sort to ensure time order before shifting or splitting
    df = df.sort_values("timestamp")
    
    df['target'] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    # Add target 1 hour ahead
    df['target_h1'] = (df[price_col].shift(-4) > df[price_col]).astype(int)
    
    return df.dropna(subset=['target', 'target_h1']).copy()


def train_and_evaluate(X, y, description):
    """Train XGBoost using true time-based split."""
    # 1. Clean cut time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Handle class imbalance safely
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    # TODO Next: Upgrade to Walk-Forward (Rolling) Validation
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Probabilities for AUC
    probs = model.predict_proba(X_test)[:, 1]
    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = 0.0
    
    print(f"\n=== {description} ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"AUC:       {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Feature importance
    importance = model.feature_importances_
    feat_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    print("\nTop features:")
    print(feat_importance.head(10))
    
    # Specific sentiment feature importance check
    sent_imp = feat_importance[feat_importance.index.str.contains("sentiment")]
    print("\nSentiment features importance:")
    print(sent_imp.head(10))
    print("===================================\n")


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
        "rss_volume", "gdelt_sentiment", "rss_sentiment", "sentiment_spike"
    ]
    
    # Ensure generated temporal sentiment features are grouped securely avoiding technical leakage
    temporal_sentiment = [c for c in all_cols if "sentiment" in c]
    sentiment_features.extend(temporal_sentiment)
    
    sentiment_cols_present = list(set([c for c in sentiment_features if c in all_cols]))
    
    # Filter text columns
    text_cols = ["title", "url", "cleaned_text", "source", "feed_source", "sentiment_label"]
    
    leakage_cols = ["future_return", "target", "target_up", "target_h1"]
    leakage_patterns = ["future", "t+1", "lead", "target"]

    def is_leakage(col):
        return (
            col in leakage_cols or
            any(p in col.lower() for p in leakage_patterns)
        )
    
    tech_features = [
        c for c in all_cols 
        if c not in metadata_cols 
        and c not in sentiment_cols_present
        and c not in text_cols
        and not is_leakage(c)
    ]
    
    # Drop any remaining non-numeric columns safely
    tech_features = [c for c in tech_features if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(df) < 10:
        print("\nWarning: Dataset is very small. Metrics will be unreliable.")
        
    for target_col in ["target", "target_h1"]:
        print(f"\n###############################################")
        print(f"       TRAINING FOR PREDICTION: {target_col.upper()}")
        print(f"###############################################")
        
        y = df[target_col]
        
        # Sanity check: correlations
        print("\nTop correlations with target:")
        if tech_features:
            corr = df[tech_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            print(corr.head(10))
        else:
            print("No technical features to correlate.")
        
        # 2. Naive Baseline (Majority Class)
        split_idx = int(len(df) * 0.8)
        y_train_base = y.iloc[:split_idx]
        y_test_base = y.iloc[split_idx:]
        
        if not y_train_base.empty:
            majority_class = y_train_base.mode()[0]
            baseline_pred = [majority_class] * len(y_test_base)
            print("\n=== Baseline (Majority Class) ===")
            print(f"Accuracy:  {accuracy_score(y_test_base, baseline_pred):.4f}")
            print(f"F1 Score:  {f1_score(y_test_base, baseline_pred, zero_division=0):.4f}")
            print("===================================\n")
            
        # Train WITH sentiment
        features_with_sentiment = tech_features + sentiment_cols_present
        X_with = df[features_with_sentiment]
        train_and_evaluate(X_with, y, "Model: WITH Sentiment Features")
        
        # Train WITHOUT sentiment
        if tech_features:
            X_without = df[tech_features]
            train_and_evaluate(X_without, y, "Model: Technical Features ONLY")
        else:
            print("No technical features available to train a secondary model.")


if __name__ == "__main__":
    main()
