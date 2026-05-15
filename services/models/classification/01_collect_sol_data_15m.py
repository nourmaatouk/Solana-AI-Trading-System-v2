from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ACTIVE_HORIZON as HORIZON_CANDLES,
    CANDLE_LIMIT_HISTORY as LIMIT_PER_CALL,
    DATA_DIR,
    EXCHANGE as EXCHANGE_NAME,
    PROCESSED_DIR,
    SYMBOL,
    TIMEFRAME,
)
from exchange_utils import fetch_ohlcv_with_retries, get_exchange, timeframe_to_milliseconds

START_DATE = "2021-01-01"
END_DATE   = None

RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FULL HISTORY FETCH
# =========================================================
def fetch_ohlcv_full_history(
    exchange,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str] = None,
    limit_per_call: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV history progressively in batches.
    """
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z") if end_date else None
    step_ms = timeframe_to_milliseconds(timeframe)

    all_rows = []
    batch_count = 0

    while True:
        batch = fetch_ohlcv_with_retries(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit_per_call,
            max_retries=MAX_RETRIES,
            retry_sleep_seconds=RETRY_SLEEP_SECONDS,
        )

        if not batch:
            print("No more data returned by exchange.")
            break

        all_rows.extend(batch)
        batch_count += 1

        first_dt = pd.to_datetime(batch[0][0], unit="ms", utc=True)
        last_dt = pd.to_datetime(batch[-1][0], unit="ms", utc=True)

        print(
            f"Batch {batch_count:04d} | rows={len(batch):03d} | "
            f"from={first_dt} | to={last_dt}"
        )

        last_ts = batch[-1][0]

        if last_ts == since:
            print("Stopping because last timestamp == since (safety stop).")
            break

        since = last_ts + step_ms

        if end_ts is not None and since >= end_ts:
            print("Reached end date.")
            break

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    if df.empty:
        raise ValueError("No data fetched.")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if end_ts is not None:
        df = df[df["timestamp"] < end_ts].copy()

    return df


# =========================================================
# FEATURES + TARGET
# =========================================================
def add_basic_features_and_target(df: pd.DataFrame, horizon_candles: int) -> pd.DataFrame:
    out = df.copy()

    # Basic returns
    out["return_1"] = out["close"].pct_change()
    out["log_return_1"] = np.log(out["close"] / out["close"].shift(1))

    # Candle structure
    out["high_low_spread"] = (out["high"] - out["low"]) / out["close"]
    out["open_close_spread"] = (out["close"] - out["open"]) / out["open"]

    # Future horizon return
    out["future_return_h"] = (out["close"].shift(-horizon_candles) / out["close"]) - 1.0

    # Binary target
    out["target_direction"] = (out["close"].shift(-horizon_candles) > out["close"]).astype(int)

    out = out.dropna().copy()
    return out


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 60)
    print("SOL/USDT 15m data collection started")
    print("=" * 60)
    print(f"Exchange           : {EXCHANGE_NAME}")
    print(f"Symbol             : {SYMBOL}")
    print(f"Timeframe          : {TIMEFRAME}")
    print(f"Start date         : {START_DATE}")
    print(f"End date           : {END_DATE}")
    print(f"Prediction horizon : {HORIZON_CANDLES} candles")
    print(f"Limit per call     : {LIMIT_PER_CALL}")
    print()

    exchange = get_exchange(EXCHANGE_NAME, load_markets=True)

    if SYMBOL not in exchange.symbols:
        raise ValueError(f"{SYMBOL} not found on {EXCHANGE_NAME}")

    raw_df = fetch_ohlcv_full_history(
        exchange=exchange,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        limit_per_call=LIMIT_PER_CALL,
    )

    processed_df = add_basic_features_and_target(
        df=raw_df,
        horizon_candles=HORIZON_CANDLES,
    )

    sym_slug = SYMBOL.replace("/", "_").lower()
    raw_csv           = RAW_DIR       / f"{sym_slug}_{TIMEFRAME}_raw.csv"
    raw_parquet       = RAW_DIR       / f"{sym_slug}_{TIMEFRAME}_raw.parquet"
    processed_csv     = PROCESSED_DIR / f"{sym_slug}_{TIMEFRAME}_direction_h{HORIZON_CANDLES}.csv"
    processed_parquet = PROCESSED_DIR / f"{sym_slug}_{TIMEFRAME}_direction_h{HORIZON_CANDLES}.parquet"

    raw_df.to_csv(raw_csv, index=False)
    raw_df.to_parquet(raw_parquet, index=False)

    processed_df.to_csv(processed_csv, index=False)
    processed_df.to_parquet(processed_parquet, index=False)

    print("\nDone.")
    print(f"Raw shape       : {raw_df.shape}")
    print(f"Processed shape : {processed_df.shape}")
    print(f"Saved raw       : {raw_csv}")
    print(f"Saved processed : {processed_csv}")

    print("\nTarget distribution:")
    print(processed_df["target_direction"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()