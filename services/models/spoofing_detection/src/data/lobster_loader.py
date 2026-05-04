"""
lobster_loader.py — LOBSTER CSV Data Loader for LOB Spoofing Detection
======================================================================

Loads LOBSTER-format data (message file + orderbook file) and returns a
merged DataFrame sorted by timestamp.  Also provides a synthetic data
generator for smoke-testing when real LOBSTER data is unavailable.

LOBSTER message file columns (no header):
    0  timestamp   — seconds since midnight (with nanosecond precision)
    1  event_type  — 1=submission, 2=partial_cancel, 3=full_cancel,
                      4=execution_visible, 5=execution_hidden,
                      6=cross_trade, 7=trading_halt
    2  order_id    — unique order identifier
    3  size        — volume of the order
    4  price       — price in dollar-price × 10,000
    5  direction   — +1 buy, −1 sell

LOBSTER orderbook file columns (no header):
    Interleaved: ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...
    For L levels → 4 × L columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column naming helpers
# ---------------------------------------------------------------------------
MESSAGE_COLS = [
    "timestamp",      # nanoseconds since midnight (float64)
    "event_type",     # 1-7 per LOBSTER spec
    "order_id",       # unique int64
    "size",           # order volume
    "price",          # raw LOBSTER price (× 10 000)
    "direction",      # +1 buy, -1 sell
]

def _orderbook_columns(n_levels: int = 10) -> list[str]:
    """Generate orderbook column names for *n_levels* depth levels.

    LOBSTER orderbook files interleave:
        ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...
    """
    cols: list[str] = []
    for i in range(1, n_levels + 1):
        cols += [
            f"ask_price_{i}",
            f"ask_vol_{i}",
            f"bid_price_{i}",
            f"bid_vol_{i}",
        ]
    return cols


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_lobster_data(
    message_path: str | Path,
    orderbook_path: str | Path,
    n_levels: int = 10,
    price_divisor: float = 10_000.0,
    filter_event_types: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load and merge LOBSTER message + orderbook files.

    Parameters
    ----------
    message_path : path to the LOBSTER message CSV file.
    orderbook_path : path to the LOBSTER orderbook CSV file.
    n_levels : number of orderbook depth levels (default 10).
    price_divisor : LOBSTER stores prices as int × 10 000.
    filter_event_types : if given, keep only these event types (1-4 typical).

    Returns
    -------
    pd.DataFrame — merged, sorted by timestamp, with columns from both files.
    """
    message_path = Path(message_path)
    orderbook_path = Path(orderbook_path)

    if not message_path.exists():
        raise FileNotFoundError(f"Message file not found: {message_path}")
    if not orderbook_path.exists():
        raise FileNotFoundError(f"Orderbook file not found: {orderbook_path}")

    # --- Load message file ---------------------------------------------------
    logger.info("Loading message file: %s", message_path)
    msg_df = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLS,
        dtype={
            "timestamp": np.float64,
            "event_type": np.int32,
            "order_id": np.int64,
            "size": np.int64,
            "price": np.int64,
            "direction": np.int32,
        },
    )

    # Convert price from LOBSTER int format to float dollars
    msg_df["price"] = msg_df["price"] / price_divisor

    # Convert timestamp from seconds-since-midnight to nanoseconds
    # LOBSTER timestamps are already in seconds with nanosecond decimal precision
    msg_df["timestamp_ns"] = (msg_df["timestamp"] * 1e9).astype(np.int64)

    # --- Load orderbook file ------------------------------------------------
    ob_cols = _orderbook_columns(n_levels)
    logger.info("Loading orderbook file: %s (%d levels)", orderbook_path, n_levels)
    ob_df = pd.read_csv(
        orderbook_path,
        header=None,
        names=ob_cols,
        dtype=np.float64,
    )

    # Convert orderbook prices from LOBSTER int format
    price_cols = [c for c in ob_cols if "price" in c]
    for col in price_cols:
        ob_df[col] = ob_df[col] / price_divisor

    # --- Validate row counts match ------------------------------------------
    if len(msg_df) != len(ob_df):
        raise ValueError(
            f"Row count mismatch: message file has {len(msg_df)} rows, "
            f"orderbook file has {len(ob_df)} rows"
        )

    # --- Merge by index (LOBSTER guarantees row-aligned files) ---------------
    merged = pd.concat([msg_df.reset_index(drop=True),
                        ob_df.reset_index(drop=True)], axis=1)

    # --- Filter event types (keep 1-4 by default) ---------------------------
    if filter_event_types is not None:
        before = len(merged)
        merged = merged[merged["event_type"].isin(filter_event_types)].copy()
        logger.info("Filtered events %s: %d → %d rows",
                     filter_event_types, before, len(merged))

    # --- Sort by timestamp (should already be sorted, but enforce) -----------
    merged.sort_values("timestamp_ns", inplace=True, kind="mergesort")
    merged.reset_index(drop=True, inplace=True)

    logger.info("Loaded %d LOB events with %d orderbook levels",
                len(merged), n_levels)
    return merged


# ---------------------------------------------------------------------------
# Synthetic data generator (smoke testing)
# ---------------------------------------------------------------------------
def generate_synthetic_lobster_data(
    n_events: int = 5_000,
    n_levels: int = 10,
    base_price: float = 150.0,
    tick_size: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic LOB data for smoke-testing the pipeline.

    Creates a DataFrame with the same schema as ``load_lobster_data`` output,
    but with parametrically generated market activity including:
    - Random order placements, cancellations, and executions
    - Realistic orderbook depth with bid/ask spread
    - Correlated price movements with mean-reversion

    Parameters
    ----------
    n_events : number of LOB events to generate.
    n_levels : orderbook depth levels (default 10).
    base_price : starting mid-price.
    tick_size : minimum price increment.
    seed : random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with the same columns as load_lobster_data output.
    """
    rng = np.random.default_rng(seed)

    # --- Timestamps (nanoseconds, ascending) --------------------------------
    # Simulate events with exponential inter-arrival times
    # Mean inter-event time ~ 50ms (typical for active LOB)
    inter_arrival_ms = rng.exponential(scale=50.0, size=n_events)
    timestamps_ms = np.cumsum(inter_arrival_ms)
    timestamp_ns = (timestamps_ms * 1e6).astype(np.int64)  # ms → ns
    timestamps_sec = timestamps_ms / 1000.0  # ms → seconds

    # --- Event types (1=submission, 2=partial_cancel, 3=full_cancel, 4=exec) -
    event_probs = [0.45, 0.10, 0.30, 0.15]  # realistic distribution
    event_types = rng.choice([1, 2, 3, 4], size=n_events, p=event_probs).astype(np.int32)

    # --- Order IDs ----------------------------------------------------------
    order_ids = np.arange(1, n_events + 1, dtype=np.int64)
    # Cancellations/executions reference prior orders
    for i in range(n_events):
        if event_types[i] in (2, 3, 4) and i > 0:
            # Reference a recent prior order
            lookback = min(i, 50)
            order_ids[i] = order_ids[i - rng.integers(1, lookback + 1)]

    # --- Directions ---------------------------------------------------------
    directions = rng.choice([-1, 1], size=n_events).astype(np.int32)

    # --- Sizes (log-normal, typical for LOB) --------------------------------
    sizes = np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n_events)).astype(np.int64)
    sizes = np.clip(sizes, 1, 50_000)

    # --- Prices (random walk around base_price) -----------------------------
    price_changes = rng.normal(0, tick_size * 0.5, size=n_events)
    # Add mean reversion
    mid_prices = np.zeros(n_events)
    mid_prices[0] = base_price
    for i in range(1, n_events):
        reversion = -0.01 * (mid_prices[i - 1] - base_price)
        mid_prices[i] = mid_prices[i - 1] + price_changes[i] + reversion

    # Round to tick size
    mid_prices = np.round(mid_prices / tick_size) * tick_size

    # Order prices: scatter around mid-price
    order_prices = mid_prices + rng.choice(
        np.arange(-5, 6) * tick_size, size=n_events
    )

    # --- Build orderbook snapshot for each event ----------------------------
    ob_data: dict[str, np.ndarray] = {}
    for level in range(1, n_levels + 1):
        spread_half = level * tick_size
        ob_data[f"ask_price_{level}"] = mid_prices + spread_half
        ob_data[f"bid_price_{level}"] = mid_prices - spread_half

        # Volumes decrease with depth (log decay)
        base_vol = rng.lognormal(mean=5.0, sigma=0.8, size=n_events)
        depth_decay = 1.0 / (1.0 + 0.3 * (level - 1))
        ob_data[f"ask_vol_{level}"] = np.round(base_vol * depth_decay).astype(np.float64)
        ob_data[f"bid_vol_{level}"] = np.round(
            base_vol * depth_decay * rng.uniform(0.7, 1.3, size=n_events)
        ).astype(np.float64)

    # --- Assemble DataFrame -------------------------------------------------
    df = pd.DataFrame({
        "timestamp": timestamps_sec,
        "event_type": event_types,
        "order_id": order_ids,
        "size": sizes,
        "price": order_prices,
        "direction": directions,
        "timestamp_ns": timestamp_ns,
    })

    for col_name, col_data in ob_data.items():
        df[col_name] = col_data

    df.sort_values("timestamp_ns", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    logger.info("Generated %d synthetic LOB events (base_price=%.2f, tick=%.4f)",
                n_events, base_price, tick_size)
    return df


def generate_sample_lobster_files(
    output_dir: str | Path = "data/raw",
    ticker: str = "AAPL",
    date: str = "2012-06-21",
    n_levels: int = 10,
    n_events: int = 1000,
):
    """
    Generates realistic synthetic LOBSTER-format CSV files as a stand-in.
    Documented as synthetic-format data for pipeline validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msg_file = output_dir / f"{ticker}_{date}_message_{n_levels}.csv"
    ob_file = output_dir / f"{ticker}_{date}_orderbook_{n_levels}.csv"

    if msg_file.exists() and ob_file.exists():
        logger.info("Sample LOBSTER files already exist at %s", output_dir)
        return msg_file, ob_file

    logger.info("Generating synthetic stand-in LOBSTER files at %s...", output_dir)
    
    # Generate using realistic distributions requested: mid ~100.0, tick 0.01, size 1-500
    df = generate_synthetic_lobster_data(
        n_events=n_events,
        n_levels=n_levels,
        base_price=100.0,
        tick_size=0.01,
        seed=42
    )

    # Save Message File (No header)
    # Time, Type, OrderID, Size, Price, Direction
    # Price must be multiplied by 10,000 for LOBSTER format
    msg_df = df[["timestamp", "event_type", "order_id", "size", "price", "direction"]].copy()
    msg_df["price"] = (msg_df["price"] * 10_000).astype(np.int64)
    msg_df.to_csv(msg_file, header=False, index=False)

    # Save Orderbook File (No header)
    # Interleaved: AskPrice1, AskSize1, BidPrice1, BidSize1, ...
    ob_cols = _orderbook_columns(n_levels)
    ob_df = df[ob_cols].copy()
    # Prices must be multiplied by 10,000
    for col in ob_df.columns:
        if "price" in col:
            ob_df[col] = (ob_df[col] * 10_000).astype(np.int64)
    
    ob_df.to_csv(ob_file, header=False, index=False)

    logger.info("Successfully created stand-in LOBSTER files (SYNTHETIC FORMAT).")
    return msg_file, ob_file


# ---------------------------------------------------------------------------
# Convenience: load from directory with LOBSTER naming convention
# ---------------------------------------------------------------------------
def load_lobster_from_directory(
    data_dir: str | Path,
    ticker: str = "SOL",
    date: str = "2024-01-02",
    n_levels: int = 10,
    filter_event_types: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load LOBSTER data using flexible naming support for samples and production files.
    
    Tries multiple naming patterns:
    1. Standard full prefix: {ticker}_{date}_34200000_57600000_message_{n_levels}.csv
    2. Sample format: {ticker}_{date}_message_{n_levels}.csv
    """
    data_dir = Path(data_dir)
    
    # Pattern 1: Standard prefix
    prefix_std = f"{ticker}_{date}_34200000_57600000"
    msg_path_std = data_dir / f"{prefix_std}_message_{n_levels}.csv"
    ob_path_std = data_dir / f"{prefix_std}_orderbook_{n_levels}.csv"

    # Pattern 2: Short sample prefix
    msg_path_short = data_dir / f"{ticker}_{date}_message_{n_levels}.csv"
    ob_path_short = data_dir / f"{ticker}_{date}_orderbook_{n_levels}.csv"

    if msg_path_std.exists() and ob_path_std.exists():
        m_path, o_path = msg_path_std, ob_path_std
    elif msg_path_short.exists() and ob_path_short.exists():
        m_path, o_path = msg_path_short, ob_path_short
    else:
        # If neither exists and we are using stand-ins, generate them
        if ticker == "AAPL" and date == "2012-06-21":
            m_path, o_path = generate_sample_lobster_files(data_dir, ticker, date, n_levels)
        else:
            raise FileNotFoundError(f"LOBSTER files for {ticker} on {date} not found in {data_dir}")

    return load_lobster_data(
        message_path=m_path,
        orderbook_path=o_path,
        n_levels=n_levels,
        filter_event_types=filter_event_types,
    )
