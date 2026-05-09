import pytest
from src.main import deduplicate_articles


def test_deduplicate_articles_keeps_unique():
    articles = [
        {"url": "http://test1.com", "title": "Article 1"},
        {"url": "http://test2.com", "title": "Article 2"},
    ]
    
    result = deduplicate_articles(articles)
    assert len(result) == 2


def test_deduplicate_articles_removes_duplicate_urls():
    articles = [
        {"url": "http://test.com", "title": "Article A"},
        {"url": "http://test.com", "title": "Article B"},  # Duplicate URL
    ]
    
    result = deduplicate_articles(articles)
    assert len(result) == 1
    assert result[0]["title"] == "Article A"


def test_deduplicate_articles_removes_duplicate_titles():
    articles = [
        {"url": "http://test1.com", "title": "Same Title"},
        {"url": "http://test2.com", "title": "Same Title"},  # Duplicate Title
    ]
    
    result = deduplicate_articles(articles)
    assert len(result) == 1
    assert result[0]["url"] == "http://test1.com"


def test_deduplicate_articles_handles_missing_fields_safely():
    articles = [
        {"url": "http://missingtitle.com"},
        {"title": "Missing URL Article"},
        {"url": "http://missingtitle.com"},  # Duplicate
        {"title": "Missing URL Article"},    # Duplicate
    ]
    
    result = deduplicate_articles(articles)
    assert len(result) == 2
