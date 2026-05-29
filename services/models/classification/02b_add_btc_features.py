"""
02b_add_btc_features.py

Downloads BTC/USDT 15m candles for the same period as SOL and merges 5
stationary BTC co-features onto the SOL features CSV.

Run AFTER 02_build_technical_features.py, BEFORE 06_merge_sentiment_features.py:

    python 02_build_technical_features.py
    python 02b_add_btc_features.py      ← this script
    python 06_merge_sentiment_features.py
    ...

BTC features added (all stationary):
    btc_return_4     — 1-hour BTC return (same horizon as production model)
    btc_return_16    — 4-hour BTC return (medium momentum)
    btc_vol_12       — 3-hour BTC return volatility
    btc_vs_sma20     — BTC price vs its 20-bar SMA (trend position)
    btc_rsi_14       — BTC RSI-14
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BTC_RAW_CSV,
    BTC_SYMBOL,
    CANDLE_LIMIT_HISTORY,
    DATA_DIR,
    EXCHANGE,
    PROCESSED_DIR,
    TECH_FEATURES_CSV,
    TIMEFRAME,
)
from exchange_utils import fetch_ohlcv_with_retries, get_exchange
from indicators import build_btc_features, compute_rsi

RAW_DIR        = DATA_DIR / "raw"
BTC_RAW_PATH   = RAW_DIR / BTC_RAW_CSV
SOL_FEAT_PATH  = PROCESSED_DIR / TECH_FEATURES_CSV


def fetch_all_ohlcv(ex, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download all 15m candles for symbol between start_ms and end_ms."""
    tf_ms    = 15 * 60 * 1_000
    all_rows: list[list] = []
    since    = start_ms

    print(f"  Downloading {symbol} from {pd.Timestamp(start_ms, unit='ms', tz='UTC')} …")

    while since < end_ms:
        batch = fetch_ohlcv_with_retries(ex, symbol, TIMEFRAME, since, CANDLE_LIMIT_HISTORY)

        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or len(batch) < CANDLE_LIMIT_HISTORY:
            break
        since = last_ts + tf_ms
        time.sleep(ex.rateLimit / 1_000)

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol}")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] <= end_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"  Downloaded {len(df):,} {symbol} candles.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("BTC CO-FEATURE BUILDER")
    print("=" * 70)

    # 1. Load SOL features to get timestamp range
    if not SOL_FEAT_PATH.exists():
        raise FileNotFoundError(f"SOL features CSV not found: {SOL_FEAT_PATH}\n"
                                "Run 02_build_technical_features.py first.")

    sol = pd.read_csv(SOL_FEAT_PATH)
    start_ms = int(sol["timestamp"].iloc[0])
    end_ms   = int(sol["timestamp"].iloc[-1])
    print(f"  SOL range : {sol['datetime'].iloc[0]}  →  {sol['datetime'].iloc[-1]}")
    print(f"  SOL rows  : {len(sol):,}")

    # 2. Download BTC (or load from cache)
    if BTC_RAW_PATH.exists():
        btc_raw = pd.read_csv(BTC_RAW_PATH)
        cached_start = int(btc_raw["timestamp"].iloc[0])
        cached_end   = int(btc_raw["timestamp"].iloc[-1])
        print(f"\n  BTC cache found: {len(btc_raw):,} rows "
              f"({btc_raw['datetime'].iloc[0]} → {btc_raw['datetime'].iloc[-1]})")

        need_refresh = cached_start > start_ms or cached_end < end_ms
        if need_refresh:
            print("  Cache does not cover full SOL range — re-downloading …")
    else:
        need_refresh = True

    if need_refresh:
        print("\n  Downloading BTC/USDT 15m data …")
        ex      = get_exchange(EXCHANGE, load_markets=True)
        btc_raw = fetch_all_ohlcv(ex, BTC_SYMBOL, start_ms, end_ms)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        btc_raw.to_csv(BTC_RAW_PATH, index=False)
        print(f"  Saved to {BTC_RAW_PATH}")
    else:
        print("  Using cached BTC data.")

    # 3. Compute BTC features
    print("\n  Computing BTC features …")
    btc_feats = build_btc_features(btc_raw)
    print(f"  BTC feature rows (after dropna): {len(btc_feats):,}")
    print(f"  Columns: {btc_feats.columns[1:].tolist()}")

    # 4. Merge onto SOL (left join — keeps all SOL rows, fills missing BTC with NaN)
    sol_merged = sol.merge(btc_feats, on="timestamp", how="left")

    btc_cols = ["btc_return_4", "btc_return_16", "btc_vol_12", "btc_vs_sma20", "btc_rsi_14"]
    n_missing = sol_merged[btc_cols].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  {n_missing:,} SOL rows have no BTC match — forward-filling …")
        sol_merged[btc_cols] = sol_merged[btc_cols].ffill()
        n_still_missing = sol_merged[btc_cols].isna().any(axis=1).sum()
        if n_still_missing > 0:
            print(f"  {n_still_missing:,} rows still NaN after ffill — dropping …")
            sol_merged = sol_merged.dropna(subset=btc_cols).reset_index(drop=True)

    # 5. Write back to the same CSV (in-place update)
    sol_merged.to_csv(SOL_FEAT_PATH, index=False)
    print(f"\n  Updated {SOL_FEAT_PATH}")
    print(f"  Final shape: {sol_merged.shape}  "
          f"(+{len(btc_cols)} BTC features, {len(sol_merged):,} rows)")
    print("\n" + "=" * 70)
    print("BTC CO-FEATURES ADDED")
    print("=" * 70)


if __name__ == "__main__":
    main()
