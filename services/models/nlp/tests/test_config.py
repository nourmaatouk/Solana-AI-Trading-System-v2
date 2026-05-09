import pytest
from src.config import Settings, get_settings


def test_settings_structure_and_defaults():
    """
    Test that the config yields appropriately structured default values 
    when initialized, ensuring specific types and fixed rules.
    """
    get_settings.cache_clear()
    settings = get_settings()
    
    # Rules specified by user:
    # RSS_FEEDS is a list
    assert isinstance(settings.RSS_FEEDS, list), "RSS_FEEDS should be loaded as a list"
    
    # GDELT_KEYWORDS is a list
    assert isinstance(settings.GDELT_KEYWORDS, list), "GDELT_KEYWORDS should be loaded as a list"
    
    # TIMEFRAME_MINUTES equals 15
    assert settings.TIMEFRAME_MINUTES == 15, "TIMEFRAME_MINUTES should default to 15"
    
    # Validation tests for types
    assert isinstance(settings.APP_ENV, str)
    assert isinstance(settings.MAX_TEXT_LENGTH, int)


def test_settings_list_parsing(monkeypatch):
    """
    Test parsing behaviour when environment variables contain comma-separated strings.
    """
    get_settings.cache_clear()
    
    # Explicity mock environment variable parsing
    monkeypatch.setenv("RSS_FEEDS", "https://feed1.com,https://feed2.com")
    monkeypatch.setenv("GDELT_KEYWORDS", "BTC,Crypto,Solana")
    
    test_settings = get_settings()
    
    assert isinstance(test_settings.RSS_FEEDS, list)
    assert len(test_settings.RSS_FEEDS) == 2
    assert "https://feed1.com" in test_settings.RSS_FEEDS
    
    assert isinstance(test_settings.GDELT_KEYWORDS, list)
    assert len(test_settings.GDELT_KEYWORDS) == 3
    assert "Crypto" in test_settings.GDELT_KEYWORDS
