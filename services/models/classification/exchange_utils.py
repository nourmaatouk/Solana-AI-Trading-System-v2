"""
Shared CCXT helpers for both historical and realtime data collection.

Centralises get_exchange, timeframe_to_milliseconds, and fetch_ohlcv_with_retries
which were previously duplicated across:
  01_collect_sol_data_15m.py, 02b_add_btc_features.py,
  realtime_market_fetcher.py, realtime_btc_fetcher.py
"""
from __future__ import annotations

import time

import ccxt

_MAX_RETRIES         = 5
_RETRY_SLEEP_SECONDS = 5


def get_exchange(exchange_name: str, *, load_markets: bool = False) -> ccxt.Exchange:
    """Create a ccxt exchange instance with rate limiting enabled.

    Parameters
    ----------
    exchange_name:
        Lower-case ccxt exchange id, e.g. ``"binance"``.
    load_markets:
        When True, calls ``exchange.load_markets()`` before returning.
        Required for scripts that look up symbols; not needed for pure OHLCV fetches.
    """
    cls = getattr(ccxt, exchange_name)
    ex  = cls({"enableRateLimit": True, "timeout": 30_000})
    if load_markets:
        ex.load_markets()
    return ex


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert a ccxt timeframe string (``"15m"``, ``"1h"``, ``"1d"``) to milliseconds."""
    unit  = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60 * 1_000
    if unit == "h":
        return value * 60 * 60 * 1_000
    if unit == "d":
        return value * 24 * 60 * 60 * 1_000
    raise ValueError(f"Unsupported timeframe unit '{unit}' in '{timeframe}'")


def fetch_ohlcv_with_retries(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int | None,
    limit: int,
    max_retries: int = _MAX_RETRIES,
    retry_sleep_seconds: int = _RETRY_SLEEP_SECONDS,
) -> list:
    """Fetch one OHLCV batch with automatic retries on network errors.

    Returns the raw list-of-lists as returned by ccxt.
    Raises the last exception if all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
        except (ccxt.RequestTimeout, ccxt.NetworkError) as exc:
            last_error = exc
            print(f"[Retry {attempt}/{max_retries}] Network/timeout error: {exc}")
            time.sleep(retry_sleep_seconds)
    raise last_error  # type: ignore[misc]
