"""
realtime_btc_fetcher.py

Fetches the latest BTC/USDT 15m candles from Binance and writes them to
    data/realtime/latest_btc_15m.csv

Run before realtime_feature_builder.py so that BTC co-features
(btc_return_4, btc_return_16, btc_vol_12, btc_vs_sma20, btc_rsi_14)
can be computed from fresh BTC data.
"""

from __future__ import annotations

import sys

from config import (
    BTC_SYMBOL,
    CANDLE_LIMIT_REALTIME as LIMIT,
    EXCHANGE as EXCHANGE_NAME,
    REALTIME_DIR,
    TIMEFRAME,
)
from exchange_utils import fetch_ohlcv_with_retries, get_exchange
from realtime_market_fetcher import fetch_latest_candles

OUTPUT_PATH = REALTIME_DIR / "latest_btc_15m.csv"


def main() -> None:
    print("=" * 60)
    print("Real-time BTC Fetcher Starting")
    print("=" * 60)

    exchange = get_exchange(EXCHANGE_NAME)
    df = fetch_latest_candles(exchange, BTC_SYMBOL, TIMEFRAME, LIMIT)

    count = len(df)
    print(f"Fetched {count} BTC candles.")
    if count < 50:
        print(f"WARNING: only {count} candles — may not be enough for btc_rsi_14 (needs 14+) or btc_return_16.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"BTC data saved to: {OUTPUT_PATH}")
    print(f"Most recent BTC candle (UTC): {df['datetime'].iloc[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
