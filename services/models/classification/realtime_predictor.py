from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "realtime" / "latest_features_with_sentiment.csv"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler_no_leak.pkl"
MODEL_PATH = BASE_DIR.parent / "models" / "xgboost_direction_model_no_leak.json"

SEQUENCE_LENGTH = 32

# Consistent with 07_prepare_no_leak_data.py
LEAKAGE_SUBSTRINGS = ["future_return", "target", "lead", "future", "t+1"]
TS_COLS = ["timestamp", "datetime", "date", "time", "ts"]


def main():
    print("=" * 60)
    print("Real-time Predictor Starting")
    print("=" * 60)

    # 1. Load data
    if not FEATURE_PATH.exists():
        print(f"Error: Features file not found: {FEATURE_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df)} rows of features.")

    if len(df) < SEQUENCE_LENGTH:
        print(f"Error: Not enough data for prediction. Need {SEQUENCE_LENGTH}, have {len(df)}.")
        sys.exit(1)

    # 2. Load expected feature columns for alignment
    import json
    feature_cols_path = BASE_DIR / "data" / "processed" / "feature_columns.json"
    if not feature_cols_path.exists():
        print(f"Error: Feature columns definition not found at {feature_cols_path}")
        sys.exit(1)
        
    with open(feature_cols_path, "r") as f:
        expected_features = json.load(f)
    
    expected_count = len(expected_features)
    current_features = [c for c in df.columns if c in expected_features]
    current_count = len(current_features)
    
    missing_features = [f for f in expected_features if f not in df.columns]
    
    print(f"Expected feature count   : {expected_count}")
    print(f"Real-time feature count : {current_count}")
    if missing_features:
        print(f"Missing features (filled with 0): {missing_features}")
    
    # 3. Align features: add missing as 0 and reorder
    for col in missing_features:
        df[col] = 0.0
        
    # Ensure correct order
    df_aligned = df[expected_features]
    
    # 4. Load model and scaler
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not SCALER_PATH.exists():
        print(f"Error: Scaler not found at {SCALER_PATH}")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    
    # 5. Prepare latest sequence
    # Take the latest 32 rows
    latest_data = df_aligned.tail(SEQUENCE_LENGTH).copy()
    
    # Get metadata from the original df (for printing)
    latest_ts = df["timestamp"].iloc[-1]
    latest_dt = df["datetime"].iloc[-1] if "datetime" in df.columns else "N/A"
    latest_close = df["close"].iloc[-1]

    X_raw = latest_data.values.astype(np.float64)
    
    # Scale
    X_scaled = scaler.transform(X_raw)
    
    # Flatten: (32, num_features) -> (1, 32 * num_features)
    X_flat = X_scaled.flatten().reshape(1, -1)
    
    # 6. Predict
    prob_up = float(model.predict_proba(X_flat)[0, 1])
    prob_down = 1.0 - prob_up
    
    signal = "UP" if prob_up > 0.5 else "DOWN"

    # 7. Output
    print("\n" + "-" * 40)
    print(f"Latest Timestamp : {latest_ts}")
    print(f"Latest UTC       : {latest_dt}")
    print(f"Latest Close     : {latest_close}")
    print("-" * 40)
    print(f"Probability UP   : {prob_up:.4f}")
    print(f"Probability DOWN : {prob_down:.4f}")
    print(f"FINAL SIGNAL     : {signal}")
    print("-" * 40)
    print("\nSuccess: Prediction completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
