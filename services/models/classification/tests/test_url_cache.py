"""
Contract tests for the URL cache TTL and rolling sentiment buffer in
realtime_sentiment_fetcher.py.

Run with (from repo root, using the classification ml_env):
    cd services/models/classification
    ml_env/bin/python -m pytest tests/test_url_cache.py -v
"""
import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Bootstrap path so the fetcher module is importable without PM2 env
_CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CLASSIFICATION_DIR))

import realtime_sentiment_fetcher as rsf


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Point CACHE_PATH at a fresh temp file for each test."""
    cache_file = tmp_path / "processed_urls.txt"
    monkeypatch.setattr(rsf, "CACHE_PATH", cache_file)
    return cache_file


def _write_raw(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n")


def _ts(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


# ── load_cached_urls ──────────────────────────────────────────────────────────

class TestLoadCachedUrls:

    def test_returns_empty_when_no_cache_file(self, tmp_cache):
        assert rsf.load_cached_urls() == set()

    def test_fresh_entry_is_returned(self, tmp_cache):
        _write_raw(tmp_cache, [f"{_ts(1)}|https://example.com/fresh"])
        urls = rsf.load_cached_urls()
        assert "https://example.com/fresh" in urls

    def test_stale_entry_is_evicted(self, tmp_cache):
        _write_raw(tmp_cache, [f"{_ts(rsf.CACHE_TTL_HOURS + 1)}|https://example.com/stale"])
        urls = rsf.load_cached_urls()
        assert "https://example.com/stale" not in urls

    def test_stale_entry_removed_from_file(self, tmp_cache):
        _write_raw(tmp_cache, [
            f"{_ts(1)}|https://example.com/fresh",
            f"{_ts(rsf.CACHE_TTL_HOURS + 1)}|https://example.com/stale",
        ])
        rsf.load_cached_urls()
        content = tmp_cache.read_text()
        assert "fresh" in content
        assert "stale" not in content

    def test_legacy_bare_url_is_recognised_but_not_rewritten(self, tmp_cache):
        _write_raw(tmp_cache, ["https://example.com/legacy"])
        urls = rsf.load_cached_urls()
        assert "https://example.com/legacy" in urls
        # Legacy lines are NOT written back (they age out naturally)
        assert "legacy" not in tmp_cache.read_text()

    def test_empty_lines_are_ignored(self, tmp_cache):
        _write_raw(tmp_cache, ["", f"{_ts(1)}|https://example.com/ok", ""])
        urls = rsf.load_cached_urls()
        assert urls == {"https://example.com/ok"}

    def test_ttl_boundary_fresh_is_kept(self, tmp_cache):
        # Exactly at TTL hours minus 1 minute → should still be fresh
        slightly_fresh = rsf.CACHE_TTL_HOURS - 1 / 60
        _write_raw(tmp_cache, [f"{_ts(slightly_fresh)}|https://example.com/boundary"])
        urls = rsf.load_cached_urls()
        assert "https://example.com/boundary" in urls

    def test_multiple_fresh_urls_all_returned(self, tmp_cache):
        lines = [f"{_ts(i)}|https://example.com/{i}" for i in range(1, 5)]
        _write_raw(tmp_cache, lines)
        urls = rsf.load_cached_urls()
        assert len(urls) == 4


# ── save_new_urls ─────────────────────────────────────────────────────────────

class TestSaveNewUrls:

    def test_saved_urls_appear_in_file(self, tmp_cache):
        rsf.save_new_urls(["https://example.com/a", "https://example.com/b"])
        content = tmp_cache.read_text()
        assert "https://example.com/a" in content
        assert "https://example.com/b" in content

    def test_saved_lines_have_timestamp_prefix(self, tmp_cache):
        rsf.save_new_urls(["https://example.com/x"])
        line = tmp_cache.read_text().strip()
        assert "|" in line, "Expected 'timestamp|url' format"
        ts_str, url = line.split("|", 1)
        # Timestamp must be parseable ISO format
        ts = datetime.fromisoformat(ts_str)
        assert url == "https://example.com/x"
        assert (datetime.now(timezone.utc) - ts).total_seconds() < 5

    def test_save_then_load_roundtrip(self, tmp_cache):
        urls = ["https://example.com/r1", "https://example.com/r2"]
        rsf.save_new_urls(urls)
        loaded = rsf.load_cached_urls()
        assert set(urls) == loaded

    def test_save_appends_to_existing(self, tmp_cache):
        rsf.save_new_urls(["https://example.com/first"])
        rsf.save_new_urls(["https://example.com/second"])
        loaded = rsf.load_cached_urls()
        assert "https://example.com/first" in loaded
        assert "https://example.com/second" in loaded


# ── Rolling sentiment buffer ──────────────────────────────────────────────────

import numpy as np
import pandas as pd


def _make_bucket(ts: str, vol: float, mean: float) -> pd.DataFrame:
    row = {col: 0.0 for col in rsf.REQUIRED_COLUMNS}
    row["timestamp"] = pd.Timestamp(ts, tz="UTC")
    row["news_volume"] = vol
    row["sentiment_mean"] = mean
    return pd.DataFrame([row])


@pytest.fixture()
def tmp_buffer(tmp_path, monkeypatch):
    monkeypatch.setattr(rsf, "BUFFER_PATH", tmp_path / "sentiment_buffer.csv")
    monkeypatch.setattr(rsf, "CACHE_PATH", tmp_path / "processed_urls.txt")


class TestSentimentBuffer:

    def test_load_buffer_empty_when_no_file(self, tmp_buffer):
        assert rsf._load_buffer().empty

    def test_append_creates_file_and_returns_row(self, tmp_buffer):
        row = _make_bucket("2026-05-15T10:00:00Z", vol=5, mean=0.3)
        buf = rsf._append_to_buffer(row)
        assert len(buf) == 1
        assert rsf.BUFFER_PATH.exists()

    def test_buffer_capped_at_lookback_buckets(self, tmp_buffer):
        for i in range(rsf.LOOKBACK_BUCKETS + 2):
            rsf._append_to_buffer(_make_bucket(f"2026-05-15T{10+i:02d}:00:00Z", vol=1, mean=0.1))
        buf = rsf._load_buffer()
        assert len(buf) == rsf.LOOKBACK_BUCKETS

    def test_aggregate_news_volume_is_sum(self, tmp_buffer):
        for i in range(3):
            rsf._append_to_buffer(_make_bucket(f"2026-05-15T{10+i:02d}:00:00Z", vol=10, mean=0.0))
        buf = rsf._load_buffer()
        agg = rsf._aggregate_buffer(buf)
        assert agg.iloc[0]["news_volume"] == 30.0

    def test_aggregate_sentiment_mean_is_decay_weighted(self, tmp_buffer):
        # Two buckets: older mean=0.0, newer mean=1.0
        rsf._append_to_buffer(_make_bucket("2026-05-15T10:00:00Z", vol=1, mean=0.0))
        rsf._append_to_buffer(_make_bucket("2026-05-15T11:00:00Z", vol=1, mean=1.0))
        buf = rsf._load_buffer()
        agg = rsf._aggregate_buffer(buf)
        # weights: [BUCKET_DECAY, 1.0], newer gets more weight → mean > 0.5
        assert agg.iloc[0]["sentiment_mean"] > 0.5

    def test_aggregate_timestamp_is_current_bucket(self, tmp_buffer):
        rsf._append_to_buffer(_make_bucket("2026-01-01T00:00:00Z", vol=1, mean=0.0))
        buf = rsf._load_buffer()
        agg = rsf._aggregate_buffer(buf)
        now_bucket = pd.Timestamp.now(tz="UTC").floor("15min")
        ts = pd.to_datetime(agg.iloc[0]["timestamp"], utc=True)
        assert abs((ts - now_bucket).total_seconds()) < 60

    def test_aggregate_empty_buffer_returns_empty(self, tmp_buffer):
        assert rsf._aggregate_buffer(pd.DataFrame()).empty

    def test_duplicate_timestamps_deduped_on_append(self, tmp_buffer):
        row = _make_bucket("2026-05-15T10:00:00Z", vol=1, mean=0.2)
        rsf._append_to_buffer(row)
        rsf._append_to_buffer(row)   # same timestamp
        buf = rsf._load_buffer()
        assert len(buf) == 1
