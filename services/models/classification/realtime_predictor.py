from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    ACTIVE_HORIZON as HORIZON,
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    ORDERBOOK_FEATURES,
    PROCESSED_DIR,
    REALTIME_DIR,
)

FEATURE_PATH         = REALTIME_DIR / "latest_features_with_sentiment.csv"
LAST_SIGNAL_PATH     = REALTIME_DIR / "last_signal.json"
FEATURE_COLUMNS_PATH = PROCESSED_DIR / "feature_columns.json"
SCALER_PATH          = PROCESSED_DIR / f"scaler_h{HORIZON}.pkl"
MODEL_PATH           = MODEL_DIR / f"xgboost_h{HORIZON}.json"


def load_expected_features(path: Path) -> list[str]:
    with open(path, "r") as f:
        features = json.load(f)

    if not isinstance(features, list) or not all(isinstance(c, str) for c in features):
        raise ValueError(f"Feature registry must be a list of strings: {path}")
    if not features:
        raise ValueError(f"Feature registry is empty: {path}")

    return features


def find_missing_features(df: pd.DataFrame, expected_features: list[str]) -> list[str]:
    return [col for col in expected_features if col not in df.columns]


def align_latest_features(
    df: pd.DataFrame,
    expected_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return the latest fully aligned 2D feature row for the flat XGBoost model."""
    missing_features = find_missing_features(df, expected_features)
    if missing_features:
        return np.empty((0, len(expected_features)), dtype=np.float64), missing_features

    latest_row = df.loc[:, expected_features].tail(1).copy()
    latest_row.replace([np.inf, -np.inf], np.nan, inplace=True)

    invalid_features = latest_row.columns[latest_row.isna().iloc[0]].tolist()
    if invalid_features:
        raise ValueError(f"Latest row has NaN/inf values in: {invalid_features}")

    return latest_row.to_numpy(dtype=np.float64), []


def model_feature_count(model: xgb.XGBClassifier) -> int:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return int(model.get_booster().num_features())


def validate_artifact_contract(
    expected_features: list[str],
    scaler,
    model: xgb.XGBClassifier,
) -> None:
    expected_count = len(expected_features)
    scaler_count = getattr(scaler, "n_features_in_", None)
    model_count = model_feature_count(model)

    errors = []
    if scaler_count is not None and int(scaler_count) != expected_count:
        errors.append(f"scaler expects {scaler_count} features, registry has {expected_count}")
    if model_count != expected_count:
        errors.append(f"model expects {model_count} features, registry has {expected_count}")

    if errors:
        raise ValueError("; ".join(errors))


def orderbook_matters(model: xgb.XGBClassifier, expected_features: list[str]) -> bool:
    """Return True only if orderbook features have non-zero importance in the loaded model.

    When the model was trained on all-zero orderbook data (WebSocket not yet
    running during the historical period), XGBoost assigns zero importance and
    the gate must not block inference — the model doesn't need orderbook data.
    """
    if not any(col in expected_features for col in ORDERBOOK_FEATURES):
        return False
    try:
        imp = model.feature_importances_
        ob_importance = sum(
            imp[expected_features.index(f)]
            for f in ORDERBOOK_FEATURES
            if f in expected_features
        )
        return ob_importance > 0.01
    except Exception:
        return False


def latest_orderbook_unavailable(df: pd.DataFrame, expected_features: list[str],
                                  model: xgb.XGBClassifier | None = None) -> bool:
    if not orderbook_matters(model, expected_features) if model is not None \
            else not any(col in expected_features for col in ORDERBOOK_FEATURES):
        return False
    if "orderbook_data_available" not in df.columns:
        return False

    return int(df["orderbook_data_available"].iloc[-1]) == 0


def score_to_signal(
    prob_up: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[str, str]:
    if prob_up >= threshold:
        return "UP", "CONFIDENT_UP"
    if prob_up <= 1.0 - threshold:
        return "DOWN", "CONFIDENT_DOWN"
    return "NEUTRAL", "LOW_CONFIDENCE"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_signal_json(
    *,
    signal: str,
    reason: str,
    status: str,
    probability_up: float | None = None,
    probability_down: float | None = None,
    latest_timestamp: Any = None,
    latest_utc: Any = None,
    latest_close: Any = None,
    details: dict[str, Any] | None = None,
    output_path: Path = LAST_SIGNAL_PATH,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "signal": signal,
        "reason": reason,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "latest_timestamp": latest_timestamp,
        "latest_utc": latest_utc,
        "latest_close": latest_close,
        "horizon": HORIZON,
        "horizon_minutes": HORIZON * 15,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "feature_path": str(FEATURE_PATH),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }
    if details:
        payload["details"] = details

    safe_payload = {key: _json_safe(value) for key, value in payload.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(safe_payload, f, indent=2)
    return safe_payload


def finish_neutral(
    reason: str,
    exit_code: int = 0,
    details: dict[str, Any] | None = None,
) -> None:
    write_signal_json(
        signal="NEUTRAL",
        reason=reason,
        status="neutral" if exit_code == 0 else "error",
        details=details,
    )
    print("-" * 40)
    print(f"FINAL SIGNAL     : NEUTRAL ({reason})")
    print(f"Signal JSON      : {LAST_SIGNAL_PATH}")
    print("-" * 40)
    print("\nFallback to neutral achieved.")
    print("=" * 60)
    sys.exit(exit_code)


def main():
    print("=" * 60)
    print("Real-time Predictor Starting")
    print("=" * 60)

    # 1. Load data
    if not FEATURE_PATH.exists():
        print(f"Error: Features file not found: {FEATURE_PATH}")
        finish_neutral("FEATURE FILE MISSING", exit_code=1, details={"missing_path": str(FEATURE_PATH)})
    
    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df)} rows of features.")

    if df.empty:
        print("Error: Feature file is empty.")
        finish_neutral("EMPTY FEATURE FILE", exit_code=1)

    # 2. Load expected feature columns for alignment
    if not FEATURE_COLUMNS_PATH.exists():
        print(f"Error: Feature columns definition not found at {FEATURE_COLUMNS_PATH}")
        finish_neutral("FEATURE REGISTRY MISSING", exit_code=1, details={"missing_path": str(FEATURE_COLUMNS_PATH)})

    try:
        expected_features = load_expected_features(FEATURE_COLUMNS_PATH)
    except ValueError as e:
        print(f"Error: {e}")
        finish_neutral("FEATURE REGISTRY ERROR", exit_code=1, details={"error": str(e)})
    
    expected_count = len(expected_features)
    current_features = [c for c in df.columns if c in expected_features]
    current_count = len(current_features)
    
    print(f"Horizon              : H{HORIZON} ({HORIZON} x 15m candles)")
    print(f"Expected feature count   : {expected_count}")
    print(f"Real-time feature count : {current_count}")

    # 3. Load model and scaler
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        finish_neutral("MODEL MISSING", exit_code=1, details={"missing_path": str(MODEL_PATH)})
    if not SCALER_PATH.exists():
        print(f"Error: Scaler not found at {SCALER_PATH}")
        finish_neutral("SCALER MISSING", exit_code=1, details={"missing_path": str(SCALER_PATH)})

    scaler = joblib.load(SCALER_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    try:
        validate_artifact_contract(expected_features, scaler, model)
    except ValueError as e:
        print(f"\nCRITICAL WARNING: Artifact contract mismatch: {e}")
        print("Model inference aborted because training artifacts do not agree.")
        finish_neutral("MODEL CONTRACT ERROR", exit_code=1, details={"error": str(e)})

    # 4. Align latest row. Missing trained features are a no-trade condition.
    try:
        X_raw, missing_features = align_latest_features(df, expected_features)
    except ValueError as e:
        print(f"\nCRITICAL WARNING: {e}")
        print("Model inference aborted to prevent invalid feature scoring.")
        finish_neutral("INVALID FEATURE VALUES", details={"error": str(e)})

    if missing_features:
        print(f"Missing features: {missing_features}")
        print(f"\nCRITICAL WARNING: Missing {len(missing_features)} trained features.")
        print("Model inference aborted to prevent random guessing.")
        finish_neutral("INSUFFICIENT FEATURES", details={"missing_features": missing_features})

    if latest_orderbook_unavailable(df, expected_features, model=model):
        print("\nCRITICAL WARNING: No matching orderbook bucket for the latest candle.")
        print("Model inference aborted because live orderbook features carry non-zero importance.")
        finish_neutral("ORDERBOOK UNAVAILABLE")
    
    # Get metadata from the original df (for printing)
    latest_ts = df["timestamp"].iloc[-1]
    latest_dt = df["datetime"].iloc[-1] if "datetime" in df.columns else "N/A"
    latest_close = df["close"].iloc[-1]
    
    # Scale
    X_scaled = scaler.transform(X_raw)

    # 5. Score the latest candle. XGBoost is a probability scorer; the gate below
    # decides whether that score is strong enough to become an actionable signal.
    prob_up = float(model.predict_proba(X_scaled)[0, 1])
    prob_down = 1.0 - prob_up
    signal, decision_reason = score_to_signal(prob_up)
    write_signal_json(
        signal=signal,
        reason=decision_reason,
        status="ok",
        probability_up=prob_up,
        probability_down=prob_down,
        latest_timestamp=latest_ts,
        latest_utc=latest_dt,
        latest_close=latest_close,
        details={
            "expected_feature_count": expected_count,
            "realtime_feature_count": current_count,
        },
    )

    # 6. Output
    print("\n" + "-" * 40)
    print(f"Latest Timestamp : {latest_ts}")
    print(f"Latest UTC       : {latest_dt}")
    print(f"Latest Close     : {latest_close}")
    print("-" * 40)
    print(f"Probability UP   : {prob_up:.4f}")
    print(f"Probability DOWN : {prob_down:.4f}")
    print(f"Decision Gate    : p(up)>={CONFIDENCE_THRESHOLD:.2f} or p(up)<={1.0 - CONFIDENCE_THRESHOLD:.2f}")
    if signal == "NEUTRAL":
        print(f"FINAL SIGNAL     : NEUTRAL ({decision_reason})")
    else:
        print(f"FINAL SIGNAL     : {signal} ({decision_reason})")
    print(f"Signal JSON      : {LAST_SIGNAL_PATH}")
    print("-" * 40)
    print("\nSuccess: Prediction completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
