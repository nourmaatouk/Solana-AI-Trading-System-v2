from __future__ import annotations

import sys

import pandas as pd

from config import (
    CANDLE_LIMIT_REALTIME as LIMIT,
    EXCHANGE as EXCHANGE_NAME,
    REALTIME_DIR,
    SYMBOL,
    TIMEFRAME,
)
from exchange_utils import fetch_ohlcv_with_retries, get_exchange

OUTPUT_PATH = REALTIME_DIR / "latest_sol_15m.csv"


def fetch_latest_candles(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    print(f"Fetching latest {limit} candles for {symbol} on {timeframe}...")
    
    try:
        ohlcv = fetch_ohlcv_with_retries(exchange, symbol, timeframe, since=None, limit=limit)
    except Exception as e:
        print(f"Error fetching data from exchange after retries: {type(e).__name__}: {e}")
        sys.exit(1)
        
    if not ohlcv:
        print("Error: No data returned from exchange.")
        sys.exit(1)
        
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    
    # Convert timestamp to UTC datetime for clarity/verification
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Sort and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def main():
    print("=" * 60)
    print("Real-time Market Fetcher Starting")
    print("=" * 60)

    exchange = get_exchange(EXCHANGE_NAME)
    
    # Fetch data
    df = fetch_latest_candles(exchange, SYMBOL, TIMEFRAME, LIMIT)
    
    # Validation
    count = len(df)
    print(f"Fetched {count} candles.")
    
    if count < 100:
        print(f"WARNING: Fewer than 100 candles fetched ({count}).")
        print("This may not be enough for technical indicators with large windows.")
    elif count < 150:
        print(f"Note: Fetched {count} candles. This meets the minimum requirement.")
    else:
        print(f"Success: Fetched {count} candles.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Latest market data saved to: {OUTPUT_PATH}")
    print(f"Most recent candle time (UTC): {df['datetime'].iloc[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
