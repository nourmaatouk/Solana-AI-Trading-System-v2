import requests
from datetime import datetime, timezone
from typing import List, Dict

from src.config import get_settings
from src.fetchers.base import BaseFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.json"


class RedditFetcher(BaseFetcher):
    """
    Fetches Solana-related posts from crypto subreddits via Reddit's public JSON API.
    No OAuth or PRAW required — uses Reddit's unauthenticated read-only endpoint.
    """

    def __init__(self):
        settings = get_settings()
        self._subreddits = settings.REDDIT_SUBREDDITS
        self._query = " OR ".join(settings.GDELT_KEYWORDS)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SolanaNewsFetcher/1.0 (research)"})

    def _fetch_subreddit(self, subreddit: str) -> List[Dict]:
        try:
            resp = self._session.get(
                _SEARCH_URL.format(subreddit=subreddit),
                params={
                    "q": self._query,
                    "sort": "new",
                    "limit": 25,
                    "t": "day",
                    "restrict_sr": 1,
                },
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning(f"Reddit rate-limited for r/{subreddit}.")
                return []
            if resp.status_code != 200:
                logger.warning(f"Reddit returned {resp.status_code} for r/{subreddit}.")
                return []

            posts = resp.json().get("data", {}).get("children", [])
            results: List[Dict] = []
            for post in posts:
                d = post.get("data", {})
                created = d.get("created_utc")
                published_at = (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created
                    else ""
                )
                results.append({
                    "source": "reddit",
                    "feed_source": f"r/{subreddit}",
                    "title": d.get("title", ""),
                    "description": d.get("selftext", ""),
                    "url": d.get("url", ""),
                    "published_at": published_at,
                })
            return results
        except Exception as e:
            logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")
            return []

    def fetch(self) -> List[Dict]:
        results: List[Dict] = []
        for subreddit in self._subreddits:
            results.extend(self._fetch_subreddit(subreddit))
        logger.info(f"Reddit: fetched {len(results)} posts from {len(self._subreddits)} subreddit(s).")
        return results
