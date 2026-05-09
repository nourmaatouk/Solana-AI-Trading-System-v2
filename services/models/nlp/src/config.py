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

    # RSS Feeds List
    RSS_FEEDS: List[str] = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cryptoslate.com/feed/",
        "https://cointelegraph.com/feed/"
    ]
    
    # GDELT Configuration
    GDELT_KEYWORDS: List[str] = ["Solana", "SOL token", "Solana blockchain"]
    GDELT_FETCH_INTERVAL_MINUTES: int = 5

    # FinBERT Settings
    MAX_TEXT_LENGTH: int = 512
    SENTIMENT_THRESHOLD: float = 0.1

    # Storage Paths and Timing
    OUTPUT_DIR: str = "data/processed"
    RAW_DIR: str = "data/raw"
    TIMEFRAME_MINUTES: int = 15

    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 3600

    # TimescaleDB
    TIMESCALE_DSN: str = "postgresql://postgres:password@localhost:5432/trading"

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
