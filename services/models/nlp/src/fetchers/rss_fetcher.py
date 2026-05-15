import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from src.config import get_settings

# Broad crypto keywords — at least one must appear in title+description
# before we bother calling FinBERT. Keeps CPU load low on generic articles.
_SOLANA_KEYWORDS = {
    "solana", "sol", "defi", "dex", "crypto", "blockchain",
    "token", "nft", "web3", "raydium", "jupiter", "serum",
    "hack", "exploit", "etf", "sec", "regulation", "whale",
    "liquidation", "stablecoin", "airdrop",
}


def _is_relevant(entry: feedparser.FeedParserDict) -> bool:
    """Returns True if the RSS entry mentions at least one crypto keyword."""
    text = " ".join([
        entry.get("title", ""),
        entry.get("description", ""),
        entry.get("summary", ""),
    ]).lower()
    return any(kw in text for kw in _SOLANA_KEYWORDS)


def _fetch_one_feed(feed_url: str) -> Tuple[List[Dict], int]:
    """Fetch and filter a single RSS feed. Returns (articles, n_skipped)."""
    results: List[Dict] = []
    skipped = 0
    try:
        feed = feedparser.parse(feed_url)
        if getattr(feed, "bozo_exception", None):
            print(f"  RSS error {feed_url}: {feed.bozo_exception}")
            return results, skipped
        feed_source = feed.feed.get("title", feed_url)
        for entry in feed.entries:
            if not _is_relevant(entry):
                skipped += 1
                continue
            results.append({
                "source":       "rss",
                "feed_source":  feed_source,
                "title":        entry.get("title", ""),
                "description":  entry.get("description", entry.get("summary", "")),
                "url":          entry.get("link", ""),
                "published_at": entry.get("published", ""),
            })
    except Exception as e:
        print(f"  RSS unexpected error {feed_url}: {e}")
    return results, skipped


class RssFetcher:
    """
    Fetches news from all configured RSS feeds concurrently (one thread per feed).
    Network I/O for all feeds overlaps, cutting total time from O(N×latency) to
    O(max_single_feed_latency).  Gracefully handles per-feed errors so a single
    slow or broken feed never stalls the pipeline.
    """

    def __init__(self):
        self.settings = get_settings()

    def fetch(self) -> List[Dict]:
        feeds = self.settings.RSS_FEEDS
        if not feeds:
            return []

        all_results: List[Dict] = []
        total_skipped = 0

        # One thread per feed — I/O-bound, GIL released during network wait
        with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
            futures = {pool.submit(_fetch_one_feed, url): url for url in feeds}
            for future in as_completed(futures):
                articles, skipped = future.result()
                all_results.extend(articles)
                total_skipped += skipped

        print(f"RSS: {len(all_results)} relevant entries kept, "
              f"{total_skipped} skipped (off-topic).")
        return all_results
