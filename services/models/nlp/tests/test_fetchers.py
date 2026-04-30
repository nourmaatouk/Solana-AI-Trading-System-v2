import pytest
import requests
import feedparser

from src.fetchers.gdelt_fetcher import GdeltFetcher
from src.fetchers.rss_fetcher import RssFetcher
from src.config import get_settings


def test_gdelt_fetcher(monkeypatch):
    """Test GDELT fetcher returns list of properly mapped articles"""
    
    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "articles": [
                    {
                        "title": "Crypto Surge",
                        "url": "http://example.com/gdelt",
                        "seendate": "20260429T100000Z",
                        "tone": 4.5
                    }
                ]
            }

    # Intercept HTTP requests dynamically
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    
    fetcher = GdeltFetcher()
    results = fetcher.fetch()
    
    assert isinstance(results, list)
    assert len(results) == 1
    
    article = results[0]
    assert article["source"] == "gdelt"
    assert "title" in article
    assert "url" in article
    assert article["url"] == "http://example.com/gdelt"
    assert "published_at" in article


def test_gdelt_fetcher_429_handling(monkeypatch):
    """Test GDELT fetcher safely handles 429 Too Many Requests."""
    class Mock429Response:
        status_code = 429
        def raise_for_status(self): pass
        
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Mock429Response())
    
    fetcher = GdeltFetcher()
    results = fetcher.fetch()
    
    assert isinstance(results, list)
    assert len(results) == 0


def test_rss_fetcher(monkeypatch):
    """Test RSS fetcher processes valid feeds and gracefully bypasses crashed feeds."""
    
    def mock_feedparser_parse(url):
        if "badfeed" in url:
            # Simulate a malformed feed
            class BadFeed:
                def __init__(self):
                    self.bozo_exception = Exception("Malformed XML")
                    self.entries = []
                    self.feed = {}
            return BadFeed()
        else:
            # Simulate a valid feed
            class ValidFeed:
                def __init__(self):
                    self.bozo_exception = None
                    self.feed = {"title": "Valid News"}
                    self.entries = [
                        {
                            "title": "Valid Headline", 
                            "link": "http://valid.com/article", 
                            "published": "2026-04-29"
                        }
                    ]
            return ValidFeed()

    # Intercept parsing function
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    
    get_settings.cache_clear()
    
    fetcher = RssFetcher()
    # Explicitly configure a bad feed and a valid feed
    fetcher.settings.RSS_FEEDS = ["http://badfeed.com", "http://validfeed.com"]
    
    results = fetcher.fetch()
    
    # Validates that bad feed did not crash the loop and 
    # the valid feed output correctly surfaced.
    assert isinstance(results, list)
    assert len(results) == 1
    
    article = results[0]
    assert article["source"] == "rss"
    assert "title" in article
    assert "url" in article
    assert article["url"] == "http://valid.com/article"
