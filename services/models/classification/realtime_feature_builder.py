from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
# Using Path(__file__).resolve().parent as base for this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m_features.csv"


# =========================================================
# INDICATOR FUNCTIONS (Copied from training pipeline)
# =========================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


# =========================================================
# FEATURE BUILDING
# =========================================================
def add_realtime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds technical features for real-time prediction.
    Must match the feature engineering used in 01_collect and 02_build_features.
    """
    out = df.copy()

    # -----------------------------------------------------
    # Basic features (from 01_collect_sol_data_15m.py)
    # -----------------------------------------------------
    out["return_1"] = out["close"].pct_change()
    out["log_return_1"] = np.log(out["close"] / out["close"].shift(1))
    out["high_low_spread"] = (out["high"] - out["low"]) / out["close"]
    out["open_close_spread"] = (out["close"] - out["open"]) / out["open"]

    # -----------------------------------------------------
    # Technical features (from 02_build_technical_features.py)
    # -----------------------------------------------------
    # Basic lagged returns
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)
    out["return_12"] = out["close"].pct_change(12)
    out["return_24"] = out["close"].pct_change(24)

    # Moving averages
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()

    out["ema_10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

    # Relative distance from averages
    out["close_vs_sma10"] = (out["close"] / (out["sma_10"] + 1e-12)) - 1.0
    out["close_vs_sma20"] = (out["close"] / (out["sma_20"] + 1e-12)) - 1.0
    out["close_vs_ema20"] = (out["close"] / (out["ema_20"] + 1e-12)) - 1.0

    # Momentum indicators
    out["rsi_14"] = compute_rsi(out["close"], period=14)
    out["rsi_7"] = compute_rsi(out["close"], period=7)

    # MACD
    macd_line, signal_line, macd_hist = compute_macd(out["close"])
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_hist

    # Bollinger Bands
    rolling_mean_20 = out["close"].rolling(20).mean()
    rolling_std_20 = out["close"].rolling(20).std()

    out["bb_mid"] = rolling_mean_20
    out["bb_upper"] = rolling_mean_20 + 2 * rolling_std_20
    out["bb_lower"] = rolling_mean_20 - 2 * rolling_std_20

    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["bb_mid"] + 1e-12)
    out["close_vs_bb_mid"] = (out["close"] / (out["bb_mid"] + 1e-12)) - 1.0

    # Volatility
    out["volatility_12"] = out["return_1"].rolling(12).std()
    out["volatility_24"] = out["return_1"].rolling(24).std()
    out["volatility_48"] = out["return_1"].rolling(48).std()

    # ATR
    out["atr_14"] = compute_atr(out, period=14)
    out["atr_pct_14"] = out["atr_14"] / (out["close"] + 1e-12)

    # Price momentum
    out["momentum_3"] = out["close"] - out["close"].shift(3)
    out["momentum_6"] = out["close"] - out["close"].shift(6)
    out["momentum_12"] = out["close"] - out["close"].shift(12)

    # Volume features
    out["volume_change_1"] = out["volume"].pct_change()
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    out["volume_vs_sma20"] = (out["volume"] / (out["volume_sma_20"] + 1e-12)) - 1.0

    # Candle behavior
    out["candle_body"] = (out["close"] - out["open"]).abs()
    out["upper_shadow"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_shadow"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_to_range"] = out["candle_body"] / ((out["high"] - out["low"]) + 1e-12)

    return out


def main():
    print("=" * 60)
    print("Real-time Feature Builder Starting")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"Error: Real-time input file not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Build features
    featured_df = add_realtime_features(df)

    # Cleaning: No NaN or inf allowed in final feature set
    featured_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before = len(featured_df)
    featured_df.dropna(inplace=True)
    n_after = len(featured_df)
    
    print(f"Cleaned NaNs/Infs: Dropped {n_before - n_after} initial padding rows.")
    
    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Features saved successfully to: {OUTPUT_PATH}")
    print(f"Final shape: {featured_df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
