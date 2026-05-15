"""
realtime_orderbook_collector.py — Layer 4: Live Orderbook Daemon

Connects to the Binance WebSocket partial-depth stream for SOL/USDT, collects
100ms top-of-book snapshots, and writes 1-minute aggregated features to:

    data/realtime/orderbook_features.csv

The CSV is consumed by realtime_merge_features.py which resamples to 15-min
buckets and merges with technical + sentiment features before inference.

Rolling window: keeps the most recent MAX_ROWS_KEPT 1-min rows (≈ 3 hours).
Status file: data/realtime/orderbook_status.json — written every save cycle
so the pipeline can detect staleness without parsing the CSV.

Run this as a persistent background process before starting the pipeline:

    python realtime_orderbook_collector.py

Or with custom symbol / output path:

    python realtime_orderbook_collector.py --symbol solusdt --top-n 20

Press Ctrl+C (or send SIGTERM) to stop cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import REALTIME_DIR

OUTPUT_PATH = REALTIME_DIR / "orderbook_features.csv"
STATUS_PATH = REALTIME_DIR / "orderbook_status.json"

# Keep at most 3 hours of 1-min rows so the CSV never grows unbounded
MAX_ROWS_KEPT = 180

# Aggregate snapshots and flush to disk every N seconds
SAVE_INTERVAL_SEC = 60

# Binance partial depth stream — 100ms top-of-book snapshots.
# Valid snapshot depths are 5, 10, and 20 levels.
WS_URL_TEMPLATE = "wss://stream.binance.com:9443/ws/{symbol}@depth{levels}@100ms"

# Reconnect: start at 1s, double each failure, cap at 60s
RECONNECT_BASE_SEC = 1
RECONNECT_MAX_SEC = 60

# How many top-of-book levels to sum for volume calculations
DEFAULT_TOP_N = 10
SUPPORTED_TOP_N = {5, 10, 20}


class OrderBookCollector:
    def __init__(
        self,
        symbol: str = "solusdt",
        top_n: int = DEFAULT_TOP_N,
        flush_interval_sec: int = SAVE_INTERVAL_SEC,
        output_path: Path = OUTPUT_PATH,
        status_path: Path = STATUS_PATH,
        max_rows_kept: int = MAX_ROWS_KEPT,
    ):
        if top_n not in SUPPORTED_TOP_N:
            raise ValueError(f"top_n must be one of {sorted(SUPPORTED_TOP_N)} for Binance partial-depth streams")
        if flush_interval_sec <= 0:
            raise ValueError("flush_interval_sec must be positive")
        if max_rows_kept <= 0:
            raise ValueError("max_rows_kept must be positive")

        self.symbol             = symbol.lower()
        self.top_n              = top_n
        self.flush_interval_sec = flush_interval_sec
        self.output_path        = output_path
        self.status_path        = status_path
        self.max_rows_kept      = max_rows_kept
        self.ws_url             = WS_URL_TEMPLATE.format(symbol=self.symbol, levels=self.top_n)
        self.rows_written_total = 0
        self.started_at_utc     = _now()
        self.last_error: str | None = None
        self.last_flush_utc: str | None = None
        self._raw: list[dict] = []   # sub-minute snapshots for current minute
        self._stop = asyncio.Event()

    # ── signal handling ───────────────────────────────────────────────────────
    def attach_signals(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop.set)
        except (NotImplementedError, RuntimeError):
            # Some event loop implementations do not support signal handlers.
            # KeyboardInterrupt is still handled around asyncio.run().
            pass

    # ── snapshot processing ───────────────────────────────────────────────────
    def _imbalance(self, bids: list, asks: list) -> float:
        bid_v = sum(float(b[1]) for b in bids[:self.top_n])
        ask_v = sum(float(a[1]) for a in asks[:self.top_n])
        total = bid_v + ask_v
        return (bid_v - ask_v) / total if total > 0 else 0.0

    def _process(self, data: dict) -> dict | None:
        bids = data.get("bids") or data.get("b", [])
        asks = data.get("asks") or data.get("a", [])
        if not bids or not asks:
            return None

        best_bid  = float(bids[0][0])
        best_ask  = float(asks[0][0])
        imbalance = self._imbalance(bids, asks)

        return {
            "timestamp":   datetime.now(timezone.utc),
            "imbalance":   imbalance,
            "spread":      best_ask - best_bid,
            "mid_price":   (best_bid + best_ask) / 2.0,
            "bid_volume":  sum(float(b[1]) for b in bids[:self.top_n]),
            "ask_volume":  sum(float(a[1]) for a in asks[:self.top_n]),
        }

    # ── 1-minute aggregation ──────────────────────────────────────────────────
    def _aggregate(self) -> pd.DataFrame | None:
        if not self._raw:
            return None

        df = pd.DataFrame(self._raw)
        df.set_index("timestamp", inplace=True)
        resampled = df.resample("1min").agg(
            imbalance_mean=("imbalance", "mean"),
            imbalance_std=("imbalance", "std"),
            spread_mean=("spread", "mean"),
            mid_price=("mid_price", "last"),
            bid_volume_mean=("bid_volume", "mean"),
            ask_volume_mean=("ask_volume", "mean"),
        ).dropna(how="all")

        resampled["pressure_indicator"] = np.where(
            resampled["imbalance_mean"] >  0.5,  1,
            np.where(resampled["imbalance_mean"] < -0.5, -1, 0),
        )

        resampled = resampled.dropna(subset=["imbalance_mean"])
        if resampled.empty:
            return None

        resampled.reset_index(inplace=True)
        resampled["timestamp"] = resampled["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        return resampled

    # ── rolling CSV write ─────────────────────────────────────────────────────
    def _flush(self) -> int:
        new_rows = self._aggregate()
        self._raw.clear()

        if new_rows is None or new_rows.empty:
            return 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing CSV, deduplicate on timestamp, keep last MAX_ROWS_KEPT
        if self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined["timestamp"] = pd.to_datetime(
            combined["timestamp"], errors="coerce", utc=True, format="mixed"
        )
        combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        combined.sort_values("timestamp", inplace=True)

        # Rolling window — keep only the latest MAX_ROWS_KEPT rows
        if len(combined) > self.max_rows_kept:
            combined = combined.tail(self.max_rows_kept).reset_index(drop=True)

        combined.to_csv(self.output_path, index=False)
        n = len(new_rows)
        self.rows_written_total += n
        self.last_flush_utc = _now()
        print(f"  [{self.last_flush_utc}] Flushed {n} row(s) → {self.output_path.name}  "
              f"(total kept: {len(combined)})")
        return n

    # ── status JSON ───────────────────────────────────────────────────────────
    def _write_status(self, *, status: str, connected: bool) -> None:
        payload = {
            "status":                     status,
            "running":                    status == "running",
            "last_update_utc":            _now(),
            "latest_orderbook_timestamp": _latest_csv_timestamp(self.output_path),
            "connected":                  connected,
            "n_rows_in_csv":              _csv_len(self.output_path),
            "rows_written_total":         self.rows_written_total,
            "last_flush_utc":             self.last_flush_utc,
            "last_error":                 self.last_error,
            "source":                     "binance_partial_depth",
            "symbol":                     self.symbol,
            "top_n":                      self.top_n,
            "websocket_url":              self.ws_url,
            "flush_interval_sec":         self.flush_interval_sec,
            "max_rows_kept":              self.max_rows_kept,
            "output_path":                str(self.output_path),
            "status_path":                str(self.status_path),
            "started_at_utc":             self.started_at_utc,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(payload, f, indent=2)

    # ── main async loop ───────────────────────────────────────────────────────
    async def run(self) -> None:
        self.attach_signals()
        retry_delay = RECONNECT_BASE_SEC

        print("=" * 60)
        print("ORDERBOOK COLLECTOR")
        print("=" * 60)
        print(f"  Source  : Binance partial depth")
        print(f"  Symbol  : {self.symbol}")
        print(f"  Top-N   : {self.top_n} levels")
        print(f"  URL     : {self.ws_url}")
        print(f"  Output  : {self.output_path}")
        print(f"  Status  : {self.status_path}")
        print(f"  Window  : {self.max_rows_kept} rows  (~{self.max_rows_kept} minutes)")
        print(f"  Flush   : every {self.flush_interval_sec}s")
        print("  Stop    : Ctrl+C or SIGTERM writes status=stopped")
        print("=" * 60)
        self._write_status(status="running", connected=False)

        while not self._stop.is_set():
            try:
                print(f"  [{_now()}] Connecting to {self.ws_url} ...")
                async with websockets.connect(self.ws_url) as ws:
                    print(f"  [{_now()}] Connected.")
                    self.last_error = None
                    retry_delay = RECONNECT_BASE_SEC
                    self._write_status(status="running", connected=True)
                    last_flush = time.monotonic()

                    async for message in ws:
                        if self._stop.is_set():
                            break

                        data     = json.loads(message)
                        snapshot = self._process(data)
                        if snapshot:
                            self._raw.append(snapshot)

                        if time.monotonic() - last_flush >= self.flush_interval_sec:
                            self._flush()
                            self._write_status(status="running", connected=True)
                            last_flush = time.monotonic()

            except (websockets.ConnectionClosed,
                    websockets.InvalidURI,
                    OSError,
                    json.JSONDecodeError,
                    ValueError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"  [{_now()}] Connection error: {exc}")

            if self._stop.is_set():
                break

            # Flush any buffered data before reconnecting
            if self._raw:
                self._flush()
            self._write_status(status="error", connected=False)

            print(f"  [{_now()}] Reconnecting in {retry_delay}s ...")
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.sleep(retry_delay)),
                    timeout=retry_delay + 1,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            retry_delay = min(retry_delay * 2, RECONNECT_MAX_SEC)

        # Final flush on clean shutdown
        if self._raw:
            self._flush()
        self._write_status(status="stopped", connected=False)
        print(f"\n  [{_now()}] Stopped cleanly.")


# ── helpers ────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_csv_timestamp(path: Path = OUTPUT_PATH) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
        if df.empty:
            return None
        latest = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed").max()
        if pd.isna(latest):
            return None
        return latest.isoformat()
    except (OSError, ValueError):
        return None


def _csv_len(path: Path = OUTPUT_PATH) -> int:
    if not path.exists():
        return 0
    try:
        return max(0, sum(1 for _ in open(path)) - 1)
    except OSError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background orderbook collector for SOL/USDT."
    )
    parser.add_argument("--symbol", default="solusdt",
                        help="Binance symbol (default: solusdt)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Top-N order book levels to sum (default: {DEFAULT_TOP_N})")
    parser.add_argument("--flush-interval", type=int, default=SAVE_INTERVAL_SEC,
                        help=f"Flush interval in seconds (default: {SAVE_INTERVAL_SEC})")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH,
                        help=f"CSV output path (default: {OUTPUT_PATH})")
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH,
                        help=f"Status JSON path (default: {STATUS_PATH})")
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS_KEPT,
                        help=f"Rolling CSV row limit (default: {MAX_ROWS_KEPT})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    coll = OrderBookCollector(
        symbol=args.symbol,
        top_n=args.top_n,
        flush_interval_sec=args.flush_interval,
        output_path=args.output_path,
        status_path=args.status_path,
        max_rows_kept=args.max_rows,
    )
    try:
        asyncio.run(coll.run())
    except KeyboardInterrupt:
        coll._write_status(status="stopped", connected=False)
        print(f"\n  [{_now()}] KeyboardInterrupt received; stopping.")
