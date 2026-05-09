from __future__ import annotations

import sys
from pathlib import Path
import time

import ccxt
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
EXCHANGE_NAME = "binance"
SYMBOL = "SOL/USDT"
TIMEFRAME = "15m"
LIMIT = 200  # Fetch 200 to ensure enough for lag features and sequencing

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m.csv"


def get_exchange(exchange_name: str):
    exchange_class = getattr(ccxt, exchange_name)
    return exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })


def fetch_latest_candles(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch the most recent candles for a given symbol and timeframe.
    """
    print(f"Fetching latest {limit} candles for {symbol} on {timeframe}...")
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching data from exchange: {e}")
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

    # Save to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Keep columns per task: timestamp, open, high, low, close, volume (and datetime as requested for UTC)
    # The user said "Keep columns: timestamp, open, high, low, close, volume"
    # But also "Ensure timestamps are UTC datetime"
    # I'll include 'datetime' as the UTC representation.
    
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Latest market data saved to: {OUTPUT_PATH}")
    print(f"Most recent candle time (UTC): {df['datetime'].iloc[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
