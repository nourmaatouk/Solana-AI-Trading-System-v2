import json
from typing import Optional, Dict

import pandas as pd

try:
    import redis as _redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "nlp:sentiment:"


class RedisCache:
    """
    Redis cache for sentiment features with graceful degradation.
    All operations are no-ops when Redis is unreachable or not installed.
    """

    def __init__(self):
        settings = get_settings()
        self._url = settings.REDIS_URL
        self._ttl = settings.REDIS_TTL_SECONDS
        self._client = None

    def _connect(self):
        if not _REDIS_AVAILABLE:
            logger.warning("redis package not installed. Cache disabled.")
            return None
        if self._client is not None:
            return self._client
        try:
            client = _redis.Redis.from_url(
                self._url, decode_responses=True, socket_connect_timeout=2
            )
            client.ping()
            self._client = client
            logger.info(f"Redis connected: {self._url}")
            return client
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Running without cache.")
            return None

    def set(self, bucket_key: str, data: Dict) -> bool:
        client = self._connect()
        if not client:
            return False
        try:
            client.setex(
                f"{_KEY_PREFIX}{bucket_key}",
                self._ttl,
                json.dumps(data, default=str),
            )
            return True
        except Exception as e:
            logger.warning(f"Redis write failed: {e}")
            return False

    def get(self, bucket_key: str) -> Optional[Dict]:
        client = self._connect()
        if not client:
            return None
        try:
            raw = client.get(f"{_KEY_PREFIX}{bucket_key}")
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"Redis read failed: {e}")
            return None

    def get_latest(self) -> Optional[Dict]:
        """Return the most recently cached sentiment bucket."""
        client = self._connect()
        if not client:
            return None
        try:
            keys = client.keys(f"{_KEY_PREFIX}*")
            if not keys:
                return None
            latest_key = max(keys)
            raw = client.get(latest_key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"Redis get_latest failed: {e}")
            return None

    def cache_dataframe(self, df: pd.DataFrame) -> int:
        """Cache every row in a sentiment DataFrame. Returns count of rows cached."""
        if df is None or df.empty:
            return 0
        count = 0
        for _, row in df.iterrows():
            key = str(row.get("timestamp", ""))
            if key and self.set(key, row.to_dict()):
                count += 1
        logger.info(f"Cached {count}/{len(df)} sentiment rows to Redis.")
        return count
