from functools import lru_cache
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings loader. 
    Loads values from environment variables or a .env file.
    Default values are provided to ensure the system works even if .env is missing.
    """
    
    # Application Config
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # RSS Feeds List — ordered by Solana relevance
    RSS_FEEDS: List[str] = [
        # Crypto-native with strong Solana coverage
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/feed/",
        "https://cryptoslate.com/feed/",
        "https://decrypt.co/feed",
        "https://theblock.co/rss.xml",
        "https://beincrypto.com/feed/",
        # Solana ecosystem
        "https://solana.com/news/rss.xml",
    ]

    # GDELT Configuration
    # lookback must be >= 15 min (GDELT update cadence); 60 min gives one full
    # candle window and avoids the frequent empty-result fallback
    GDELT_KEYWORDS: List[str] = [
        "Solana", "SOL token", "Solana blockchain",
        "SOL price", "Solana DeFi", "Jupiter exchange",
        "Raydium", "Solana ETF",
    ]
    GDELT_FETCH_INTERVAL_MINUTES: int = 60

    # FinBERT Settings
    MAX_TEXT_LENGTH: int = 512
    SENTIMENT_THRESHOLD: float = 0.1

    # Storage Paths and Timing
    OUTPUT_DIR: str = "data/processed"
    RAW_DIR: str = "data/raw"
    TIMEFRAME_MINUTES: int = 15

    # Redis Cache — override in production via REDIS_URL env var
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 3600

    # TimescaleDB — set TIMESCALE_DSN env var in production; no default credentials
    TIMESCALE_DSN: str = ""

    # Reddit Fetcher
    REDDIT_SUBREDDITS: List[str] = ["solana", "CryptoCurrency", "CryptoMarkets"]
    REDDIT_FETCH_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False
    )

    @field_validator("RSS_FEEDS", "GDELT_KEYWORDS", "REDDIT_SUBREDDITS", mode="before")
    def split_comma_separated_strings(cls, value: Union[str, List[str]]) -> List[str]:
        """Convert comma-separated strings from env vars into lists."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings.
    lru_cache prevents reloading the .env file and re-validating upon multiple calls.
    """
    return Settings()
