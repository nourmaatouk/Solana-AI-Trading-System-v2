from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from indicators import add_time_features, compute_atr, compute_macd, compute_rsi


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sol_usdt_15m_direction_h4.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sol_usdt_15m_features_h4.csv"


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # -----------------------------------------------------
    # Time-of-day / day-of-week (cyclical encoding)
    # -----------------------------------------------------
    out = add_time_features(out)

    # -----------------------------------------------------
    # Basic lagged returns
    # -----------------------------------------------------
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)
    out["return_12"] = out["close"].pct_change(12)
    out["return_24"] = out["close"].pct_change(24)

    # -----------------------------------------------------
    # Moving averages
    # -----------------------------------------------------
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()

    out["ema_10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

    # Relative distance from averages
    out["close_vs_sma10"] = (out["close"] / out["sma_10"]) - 1.0
    out["close_vs_sma20"] = (out["close"] / out["sma_20"]) - 1.0
    out["close_vs_ema20"] = (out["close"] / out["ema_20"]) - 1.0

    # -----------------------------------------------------
    # Momentum indicators
    # -----------------------------------------------------
    out["rsi_14"] = compute_rsi(out["close"], period=14)
    out["rsi_7"] = compute_rsi(out["close"], period=7)

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------
    macd_line, signal_line, macd_hist = compute_macd(out["close"])
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_hist
    # Stationary MACD — normalised by close price
    out["macd_line_pct"]   = macd_line   / (out["close"] + 1e-12)
    out["macd_signal_pct"] = signal_line / (out["close"] + 1e-12)
    out["macd_hist_pct"]   = macd_hist   / (out["close"] + 1e-12)

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------
    rolling_mean_20 = out["close"].rolling(20).mean()
    rolling_std_20 = out["close"].rolling(20).std()

    out["bb_mid"] = rolling_mean_20
    out["bb_upper"] = rolling_mean_20 + 2 * rolling_std_20
    out["bb_lower"] = rolling_mean_20 - 2 * rolling_std_20

    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["bb_mid"] + 1e-12)
    out["close_vs_bb_mid"] = (out["close"] / (out["bb_mid"] + 1e-12)) - 1.0

    # -----------------------------------------------------
    # Volatility
    # -----------------------------------------------------
    out["volatility_12"] = out["return_1"].rolling(12).std()
    out["volatility_24"] = out["return_1"].rolling(24).std()
    out["volatility_48"] = out["return_1"].rolling(48).std()

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------
    out["atr_14"] = compute_atr(out, period=14)
    out["atr_pct_14"] = out["atr_14"] / (out["close"] + 1e-12)

    # -----------------------------------------------------
    # Price momentum
    # -----------------------------------------------------
    out["momentum_3"] = out["close"] - out["close"].shift(3)
    out["momentum_6"] = out["close"] - out["close"].shift(6)
    out["momentum_12"] = out["close"] - out["close"].shift(12)

    # -----------------------------------------------------
    # Regime & Stationary Z-Scores (Alpha Boost)
    # -----------------------------------------------------
    rolling_mean_72 = out["close"].rolling(72).mean()
    rolling_std_72 = out["close"].rolling(72).std()
    out["z_score_72"] = (out["close"] - rolling_mean_72) / (rolling_std_72 + 1e-12)
    
    rolling_vol_mean_72 = out["volume"].rolling(72).mean()
    rolling_vol_std_72 = out["volume"].rolling(72).std()
    out["volume_z_score_72"] = (out["volume"] - rolling_vol_mean_72) / (rolling_vol_std_72 + 1e-12)
    
    # Is short term momentum aligned with long term trend?
    out["trend_alignment"] = (out["ema_10"] - out["sma_50"]) / (out["close"] + 1e-12)

    # -----------------------------------------------------
    # Volume features
    # -----------------------------------------------------
    out["volume_change_1"] = out["volume"].pct_change()
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    out["volume_vs_sma20"] = (out["volume"] / (out["volume_sma_20"] + 1e-12)) - 1.0

    # -----------------------------------------------------
    # Candle behavior
    # -----------------------------------------------------
    out["candle_body"] = (out["close"] - out["open"]).abs()
    out["upper_shadow"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_shadow"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["body_to_range"]    = out["candle_body"]  / ((out["high"] - out["low"]) + 1e-12)
    # Stationary shadow sizes — normalised by close price
    out["upper_shadow_pct"] = out["upper_shadow"] / (out["close"] + 1e-12)
    out["lower_shadow_pct"] = out["lower_shadow"] / (out["close"] + 1e-12)

    # Remove rows with NaN caused by rolling windows
    out = out.dropna().reset_index(drop=True)

    return out


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 60)
    print("Building technical features")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded input shape: {df.shape}")

    featured_df = add_technical_features(df)

    featured_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved features to: {OUTPUT_PATH}")
    print(f"Final shape: {featured_df.shape}")
    print("\nColumns:")
    print(featured_df.columns.tolist())

    print("\nPreview:")
    print(featured_df.head())


if __name__ == "__main__":
    main()