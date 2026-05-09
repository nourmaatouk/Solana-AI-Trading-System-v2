"""
train_sentiment_inference.py

Trains a dedicated NLP probabilistic inference model using Logistic Regression.
Extracts targets chronologically matched to the Macro Classification model to 
ensure the Fusion Layer synthesizes correctly without contradiction.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Determine paths securely
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] # Points to `services/models`

TECH_CSV_PATH = PROJECT_ROOT / "classification" / "data" / "processed" / "sol_usdt_15m_features_h4.csv"
SENTIMENT_CSV_PATH = PROJECT_ROOT / "nlp" / "data" / "processed" / "sentiment_features_15m.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print(" NLP INFERENCE TRAINING MODULE (H4 HORIZON)")
    print("=" * 60)

    # 1. LOAD DATA
    if not TECH_CSV_PATH.exists():
        print(f"ERROR: Cannot find Macro features at {TECH_CSV_PATH}")
        sys.exit(1)
    if not SENTIMENT_CSV_PATH.exists():
        print(f"ERROR: Cannot find Sentiment dataset at {SENTIMENT_CSV_PATH}")
        sys.exit(1)

    print("Loading technical targets...")
    # Find timestamp col gracefully just like in 06
    df_tech = pd.read_csv(TECH_CSV_PATH)
    ts_tech = [c for c in df_tech.columns if c.lower() in ["timestamp", "datetime", "date", "time"]][0]
    df_tech[ts_tech] = pd.to_datetime(df_tech[ts_tech], errors="coerce", utc=True)
    df_tech = df_tech.sort_values(ts_tech).set_index(ts_tech)

    print("Loading sentiment features...")
    df_sent = pd.read_csv(SENTIMENT_CSV_PATH)
    ts_sent = [c for c in df_sent.columns if c.lower() in ["timestamp", "datetime", "date", "time"]][0]
    df_sent[ts_sent] = pd.to_datetime(df_sent[ts_sent], errors="coerce", utc=True)
    df_sent = df_sent.sort_values(ts_sent).set_index(ts_sent)

    # 2. FEATURE ENGINEERING (Rolling Aggregation)
    print("Aggregating sentiment dynamically to 1-Hour lookback windows...")
    df_sent_sub = df_sent[['sentiment_mean', 'news_volume']].copy()

    # Join left on tech index to ensure absolute synchronicity with macro
    df_master = pd.DataFrame(index=df_tech.index)
    df_master = df_master.join(df_sent_sub, how='left')

    # If no news initially, volume is 0
    df_master['news_volume'] = df_master['news_volume'].fillna(0)
    
    # Forward fill sentiment mean up to 4 intervals (1H memory) if news dies out
    # If it is totally empty, fill with 0 (neutral sentiment)
    df_master['sentiment_mean_1h'] = df_master['sentiment_mean'].ffill(limit=4).fillna(0.0)
    
    # Sum volume over 4 intervals explicitly
    df_master['news_volume_1h'] = df_master['news_volume'].rolling(window=4, min_periods=1).sum().fillna(0)

    # 3. TARGET GENERATION (Matched to Classification Filter)
    horizon = 4 # Predicting 1 Hour Ahead (H4 at 15m)
    df_master['close'] = df_tech['close']
    df_master["future_return"] = df_master["close"].pct_change(horizon).shift(-horizon)

    df_master["target_direction"] = np.nan
    df_master.loc[df_master["future_return"] > 0.005, "target_direction"] = 1
    df_master.loc[df_master["future_return"] < -0.005, "target_direction"] = 0

    n_before = len(df_master)
    df_master.dropna(subset=["target_direction"], inplace=True)
    df_master["target_direction"] = df_master["target_direction"].astype(int)
    print(f"Dropped {n_before - len(df_master)} noise rows inside the [-0.5%, 0.5%] chop bracket.")

    X = df_master[['sentiment_mean_1h', 'news_volume_1h']].values
    y = df_master['target_direction'].values

    # 4. CHRONOLOGICAL SPLITTING
    train_end = int(len(X) * 0.85) # 85% Train matching
    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class Distribution: {dict(zip(unique, counts))}")

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Protect against datasets so tiny they only have 1 class target available
    if len(unique) < 2:
        print("\nCRITICAL WARNING: The sentiment dataset slice only possesses a single target class. "
              "Logistic Regression cannot be trained. Your sentiment dataset may be exceptionally tiny right now.")
        sys.exit(0)

    # 5. MODEL TRAINING
    print("\nTraining Logistic Regression inference head...")
    model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 6. EVALUATION
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    
    print("\n--- RESULTS OUT-OF-SAMPLE ---")
    print(f"AUC Score: {auc:.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Print Coefficients
    print(f"Learned Coefficients -> Sentiment Polarity: {model.coef_[0][0]:.4f} | News Volume: {model.coef_[0][1]:.4f}")

    # 7. SERIALIZATION
    model_path = MODEL_DIR / "nlp_inference.pkl"
    scaler_path = MODEL_DIR / "nlp_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSuccessfully deployed Model to: {model_path}")
    print(f"Successfully deployed Scaler to: {scaler_path}")

if __name__ == "__main__":
    main()
