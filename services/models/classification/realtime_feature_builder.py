from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import compute_atr, compute_macd, compute_rsi


# =========================================================
# CONFIG
# =========================================================
BASE_DIR    = Path(__file__).resolve().parent
INPUT_PATH  = BASE_DIR / "data" / "realtime" / "latest_sol_15m.csv"
BTC_PATH    = BASE_DIR / "data" / "realtime" / "latest_btc_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m_features.csv"


# =========================================================
# FEATURE BUILDING
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"], utc=True)
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek
    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    return df


def add_realtime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds technical features for real-time prediction.
    Must match the feature engineering used in 01_collect and 02_build_features.
    """
    out = df.copy()

    # -----------------------------------------------------
    # Time-of-day / day-of-week (cyclical encoding)
    # -----------------------------------------------------
    out = add_time_features(out)

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
    out["macd_line_pct"]   = macd_line   / (out["close"] + 1e-12)
    out["macd_signal_pct"] = signal_line / (out["close"] + 1e-12)
    out["macd_hist_pct"]   = macd_hist   / (out["close"] + 1e-12)

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

    # Volume features
    out["volume_change_1"] = out["volume"].pct_change()
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    out["volume_vs_sma20"] = (out["volume"] / (out["volume_sma_20"] + 1e-12)) - 1.0

    # Candle behavior
    out["candle_body"] = (out["close"] - out["open"]).abs()
    out["upper_shadow"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_shadow"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_to_range"]    = out["candle_body"]  / ((out["high"] - out["low"]) + 1e-12)
    out["upper_shadow_pct"] = out["upper_shadow"] / (out["close"] + 1e-12)
    out["lower_shadow_pct"] = out["lower_shadow"] / (out["close"] + 1e-12)

    return out


# =========================================================
# BTC CO-FEATURES
# =========================================================
def compute_btc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs       = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def build_btc_features(btc: pd.DataFrame) -> pd.DataFrame:
    ret1  = btc["close"].pct_change()
    sma20 = btc["close"].rolling(20).mean()
    feats = pd.DataFrame({
        "timestamp":     btc["timestamp"],
        "btc_return_4":  btc["close"].pct_change(4),
        "btc_return_16": btc["close"].pct_change(16),
        "btc_vol_12":    ret1.rolling(12).std(),
        "btc_vs_sma20":  (btc["close"] / (sma20 + 1e-12)) - 1.0,
        "btc_rsi_14":    compute_btc_rsi(btc["close"]),
    })
    return feats.dropna(subset=["btc_return_16", "btc_rsi_14"])


def merge_btc_features(sol_df: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC co-features from latest_btc_15m.csv and left-merge onto sol_df."""
    btc_cols = ["btc_return_4", "btc_return_16", "btc_vol_12", "btc_vs_sma20", "btc_rsi_14"]

    if not BTC_PATH.exists():
        print(f"  WARNING: BTC realtime file not found ({BTC_PATH}). "
              "BTC features will be zero-filled. Run realtime_btc_fetcher.py first.")
        for col in btc_cols:
            sol_df[col] = 0.0
        return sol_df

    btc_raw  = pd.read_csv(BTC_PATH)
    btc_feat = build_btc_features(btc_raw)

    merged = sol_df.merge(btc_feat, on="timestamp", how="left")

    n_missing = merged[btc_cols].isna().any(axis=1).sum()
    if n_missing > 0:
        merged[btc_cols] = merged[btc_cols].ffill().bfill()
        still_missing = merged[btc_cols].isna().any(axis=1).sum()
        if still_missing > 0:
            print(f"  WARNING: {still_missing} rows have no BTC match even after fill — zero-filling.")
            merged[btc_cols] = merged[btc_cols].fillna(0.0)

    return merged


def main():
    print("=" * 60)
    print("Real-time Feature Builder Starting")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"Error: Real-time input file not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Build SOL technical features
    featured_df = add_realtime_features(df)

    # Merge BTC co-features (btc_return_4, btc_return_16, btc_vol_12, btc_vs_sma20, btc_rsi_14)
    print("Merging BTC co-features …")
    featured_df = merge_btc_features(featured_df)

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
