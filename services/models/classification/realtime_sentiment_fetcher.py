from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add the parent nlp directory to sys.path to allow importing from src
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
NLP_DIR = PROJECT_ROOT / "nlp"
sys.path.append(str(NLP_DIR))

try:
    from src.contracts.layer2_contract import normalize_layer2_bucket_frame
except ImportError as e:
    raise ImportError(
        "Could not import NLP layer-2 contracts. Ensure services/models/nlp is "
        "available on PYTHONPATH."
    ) from e

_NLP_IMPORT_ERROR: ImportError | None = None
try:
    from src.fetchers.gdelt_fetcher import GdeltFetcher
    from src.fetchers.rss_fetcher import RssFetcher
    from src.processing.sentiment_pipeline import process_articles
    from src.models.finbert_wrapper import FinBertSentimentAnalyzer
    from src.aggregation.feature_aggregator import aggregate_sentiment_15m
except ImportError as e:
    # The fallback/decay helpers only need the contract normalizer above. Keep
    # them importable in lightweight CI environments that do not install the
    # full FinBERT/transformers stack; main() fails clearly if called.
    _NLP_IMPORT_ERROR = e

# =========================================================
# CONFIG
# =========================================================
OUTPUT_PATH  = BASE_DIR / "data" / "realtime" / "latest_sentiment_15m.csv"
CACHE_PATH   = BASE_DIR / "data" / "realtime" / "processed_urls.txt"
BUFFER_PATH  = BASE_DIR / "data" / "realtime" / "sentiment_buffer.csv"

REQUIRED_COLUMNS = [
    "timestamp", "sentiment_mean", "sentiment_std", "news_volume",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "strong_positive_ratio", "strong_negative_ratio",
    "max_confidence", "event_count", "negative_event_count",
    "positive_event_count", "high_impact_event_count", "high_impact_event",
    "relevance_weighted_sentiment", "event_signal_mean", "event_signal_sum",
    "gdelt_volume", "rss_volume", "gdelt_sentiment", "rss_sentiment",
    "sentiment_momentum", "news_volume_change", "sentiment_spike",
    "gdelt_tone_mean",
]

CACHE_TTL_HOURS = 24    # URLs older than this are evicted so recurring articles re-enter
LOOKBACK_BUCKETS = 4    # rolling window = 4 × 15 min = 1 hour
BUCKET_DECAY     = 0.7  # weight multiplier per older bucket (most recent = 1.0)

def create_zero_row() -> pd.DataFrame:
    """Creates a fallback row with all sentiment metrics set to zero."""
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    data = {col: [0.0] for col in REQUIRED_COLUMNS}
    data["timestamp"] = [now]
    return normalize_layer2_bucket_frame(pd.DataFrame(data))


def load_cached_urls() -> set:
    """
    Reads the URL cache, drops entries older than CACHE_TTL_HOURS, rewrites
    the file with only fresh entries, and returns the set of valid URLs.
    Format per line: ISO-timestamp|url  (legacy bare-url lines are kept but
    treated as ageless so they get evicted on the next successful write cycle).
    """
    if not CACHE_PATH.exists():
        return set()

    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)
    fresh_lines: List[str] = []
    valid_urls: set = set()

    with open(CACHE_PATH, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "|" in line:
                ts_str, url = line.split("|", 1)
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff:
                        fresh_lines.append(line)
                        valid_urls.add(url)
                    # else: entry is stale — drop it
                except ValueError:
                    # Malformed timestamp — keep as legacy
                    fresh_lines.append(line)
                    valid_urls.add(url)
            else:
                # Legacy bare-url line — count as seen but let it age out
                # naturally (not written back, so gone after next prune cycle)
                valid_urls.add(line)

    # Rewrite with only fresh entries (pruning stale ones)
    with open(CACHE_PATH, "w") as f:
        for line in fresh_lines:
            f.write(line + "\n")

    return valid_urls


def save_new_urls(new_urls: List[str]):
    from datetime import datetime, timezone
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(CACHE_PATH, "a") as f:
        for url in new_urls:
            f.write(f"{now}|{url}\n")

# ── Rolling sentiment buffer ──────────────────────────────────────────────────
# Columns weighted by time-decay (most recent bucket = 1.0).
_WEIGHTED_COLS = [
    "sentiment_mean", "sentiment_std",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "strong_positive_ratio", "strong_negative_ratio",
    "relevance_weighted_sentiment", "event_signal_mean",
    "gdelt_sentiment", "rss_sentiment", "gdelt_tone_mean",
    "sentiment_spike", "max_confidence",
]
# Columns summed across the window (total activity counts).
_SUM_COLS = [
    "news_volume", "gdelt_volume", "rss_volume",
    "event_count", "negative_event_count", "positive_event_count",
    "high_impact_event_count", "event_signal_sum",
]


def _load_buffer() -> pd.DataFrame:
    if not BUFFER_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(BUFFER_PATH)
        return df.tail(LOOKBACK_BUCKETS) if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _append_to_buffer(row: pd.DataFrame) -> pd.DataFrame:
    """Append one bucket row to the rolling buffer, prune to LOOKBACK_BUCKETS, persist."""
    existing = _load_buffer()
    combined = pd.concat([existing, row], ignore_index=True) if not existing.empty else row.copy()
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce", utc=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp")
    combined = combined.tail(LOOKBACK_BUCKETS).reset_index(drop=True)
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BUFFER_PATH, index=False)
    return combined


def _aggregate_buffer(buffer: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse LOOKBACK_BUCKETS rows into one weighted row.

    Weights decay exponentially from newest (1.0) to oldest (BUCKET_DECAY^(n-1)).
    Continuous sentiment signals are weighted-averaged; volume/count columns are
    summed so the model sees total activity over the full window.
    The output timestamp is always the current 15-min floor so the pipeline
    never receives a stale timestamp.
    """
    if buffer.empty:
        return pd.DataFrame()

    df = buffer.copy()
    n = len(df)
    weights = np.array([BUCKET_DECAY ** (n - 1 - i) for i in range(n)], dtype=float)
    w_sum = weights.sum()

    # Start from the latest row so non-aggregated fields (labels, flags) are current.
    result = df.iloc[-1:].copy().reset_index(drop=True)

    for col in _WEIGHTED_COLS:
        if col in df.columns:
            result[col] = float((df[col].fillna(0.0).values * weights).sum() / w_sum)

    for col in _SUM_COLS:
        if col in df.columns:
            result[col] = float(df[col].fillna(0.0).sum())

    if "high_impact_event" in df.columns:
        result["high_impact_event"] = int(df["high_impact_event"].max())

    # Always stamp with the current bucket so downstream joins stay fresh.
    result["timestamp"] = pd.Timestamp.now(tz="UTC").floor("15min")

    return result


def _from_buffer_or_zero(reason: str) -> pd.DataFrame:
    """Return a decay-weighted aggregate of the rolling buffer, or a zero row if empty."""
    print(f"{reason} Using rolling buffer.")
    buf = _load_buffer()
    if buf.empty:
        print("  Buffer empty — outputting zero row.")
        return create_zero_row()
    return normalize_layer2_bucket_frame(_aggregate_buffer(buf))


def main():
    print("=" * 60)
    print("Real-time Sentiment Fetcher Starting")
    print("=" * 60)

    if _NLP_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing optional NLP runtime dependency/module required for live "
            f"sentiment fetching: {_NLP_IMPORT_ERROR}"
        )

    try:
        # Load cache
        seen_urls = load_cached_urls()
        print(f"Loaded {len(seen_urls)} cached URLs.")

        # 1. Fetch GDELT and all RSS feeds in parallel
        print("Fetching GDELT and RSS articles (parallel)...")
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            gdelt_future = pool.submit(GdeltFetcher().fetch)
            rss_future   = pool.submit(RssFetcher().fetch)
            gdelt_articles = gdelt_future.result()
            rss_articles   = rss_future.result()
        elapsed = time.monotonic() - t0
        print(f"  Fetch done in {elapsed:.1f}s  "
              f"(GDELT={len(gdelt_articles)}  RSS={len(rss_articles)})")
        all_articles = gdelt_articles + rss_articles
        
        # Deduplicate
        fresh_articles = [a for a in all_articles if a.get("url") not in seen_urls]
        print(f"Found {len(all_articles)} total articles. {len(fresh_articles)} are new.")
        
        if not fresh_articles:
            print("No new articles found this cycle.")
            df_final = _from_buffer_or_zero("No fresh articles.")
        else:
            # Cache all fresh URLs immediately — relevant or not — so off-topic
            # articles are never re-evaluated on the next cycle.
            save_new_urls([a["url"] for a in fresh_articles if a.get("url")])

            print(f"Processing {len(fresh_articles)} new articles with FinBERT...")

            # 2. Extract sentiment — relevance filter runs before FinBERT inside process_articles.
            analyzer = FinBertSentimentAnalyzer()
            df_processed = process_articles(fresh_articles, analyzer=analyzer, relevance_threshold=1.5)

            if df_processed.empty:
                print("No relevant articles after filtering.")
                df_final = _from_buffer_or_zero("No relevant articles.")
            else:
                # 3. Stamp all articles with the current 15-min bucket so they
                # aggregate into one row (not split by their RSS publish times).
                print("Aggregating sentiment into 15-minute buckets...")
                df_processed["timestamp"] = pd.Timestamp.now(tz="UTC").floor("15min")
                df_aggregated = aggregate_sentiment_15m(df_processed)

                if df_aggregated.empty:
                    df_final = _from_buffer_or_zero("Aggregation empty.")
                else:
                    latest_bucket = normalize_layer2_bucket_frame(
                        df_aggregated.sort_values("timestamp").tail(1).copy()
                    )
                    # 4. Append to rolling buffer and aggregate across LOOKBACK_BUCKETS.
                    buffer = _append_to_buffer(latest_bucket)
                    agg = _aggregate_buffer(buffer)
                    df_final = normalize_layer2_bucket_frame(agg)
                    print(f"  Buffer depth: {len(buffer)} bucket(s)  "
                          f"(window = {len(buffer) * 15} min)")

        # 4. Final Format Check
        for col in REQUIRED_COLUMNS:
            if col not in df_final.columns:
                df_final[col] = 0.0
        
        df_final = normalize_layer2_bucket_frame(df_final)[REQUIRED_COLUMNS]

        # 5. Save Output
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(OUTPUT_PATH, index=False)
        
        print(f"Real-time sentiment saved to: {OUTPUT_PATH}")
        print(f"Timestamp: {df_final['timestamp'].iloc[0]}")
        print(f"Mean Sentiment: {df_final['sentiment_mean'].iloc[0]:.4f}")
        print(f"Volume: {df_final['news_volume'].iloc[0]}")
        
    except Exception as e:
        print(f"Error in sentiment fetcher: {e}")
        df_final = create_zero_row()
        df_final.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)

if __name__ == "__main__":
    main()
