import feedparser
from typing import List, Dict
from src.config import get_settings


class RssFetcher:
    """
    Fetches news from standard RSS feeds configured via the environment.
    Gracefully handles network errors to prevent one missing feed from breaking the pipeline.
    """
    
    def __init__(self):
        self.settings = get_settings()

    def fetch(self) -> List[Dict]:
        results = []
        
        for feed_url in self.settings.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                
                # Prevent parsing errors from contaminating the batch
                if getattr(feed, "bozo_exception", None):
                    print(f"Error parsing feed {feed_url}: {feed.bozo_exception}")
                    continue
                    
                feed_source = feed.feed.get("title", feed_url)
                
                for entry in feed.entries:
                    results.append({
                        "source": "rss",
                        "feed_source": feed_source,
                        "title": entry.get("title", ""),
                        "description": entry.get("description", entry.get("summary", "")),
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", "")
                    })
            except Exception as e:
                print(f"Unexpected error formatting RSS feed {feed_url}: {e}")
                continue

        return results
