import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TABLE = "sentiment_features_15m"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    time                         TIMESTAMPTZ NOT NULL,
    sentiment_mean               DOUBLE PRECISION,
    sentiment_std                DOUBLE PRECISION,
    news_volume                  INTEGER,
    positive_ratio               DOUBLE PRECISION,
    negative_ratio               DOUBLE PRECISION,
    neutral_ratio                DOUBLE PRECISION,
    strong_positive_ratio        DOUBLE PRECISION,
    strong_negative_ratio        DOUBLE PRECISION,
    max_confidence               DOUBLE PRECISION,
    event_count                  INTEGER,
    high_impact_event_count      INTEGER,
    high_impact_event            INTEGER,
    event_signal_sum             DOUBLE PRECISION,
    relevance_weighted_sentiment DOUBLE PRECISION,
    gdelt_volume                 INTEGER,
    rss_volume                   INTEGER,
    gdelt_sentiment              DOUBLE PRECISION,
    rss_sentiment                DOUBLE PRECISION,
    sentiment_momentum           DOUBLE PRECISION,
    sentiment_spike              DOUBLE PRECISION
);
"""

_CREATE_HYPERTABLE_SQL = f"SELECT create_hypertable('{_TABLE}', 'time', if_not_exists => TRUE);"

# Columns in insert order (timestamp → 'time' in DB)
_INSERT_COLS = [
    "timestamp", "sentiment_mean", "sentiment_std", "news_volume",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "strong_positive_ratio", "strong_negative_ratio", "max_confidence",
    "event_count", "high_impact_event_count", "high_impact_event",
    "event_signal_sum", "relevance_weighted_sentiment",
    "gdelt_volume", "rss_volume", "gdelt_sentiment", "rss_sentiment",
    "sentiment_momentum", "sentiment_spike",
]


class TimescaleWriter:
    """Writes aggregated sentiment features to TimescaleDB with graceful degradation."""

    def __init__(self):
        settings = get_settings()
        self._dsn = settings.TIMESCALE_DSN
        self._conn = None

    def _connect(self):
        if not _PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not installed. TimescaleDB writes disabled.")
            return None
        if self._conn and not self._conn.closed:
            return self._conn
        try:
            self._conn = psycopg2.connect(self._dsn)
            logger.info("TimescaleDB connected.")
            return self._conn
        except Exception as e:
            logger.warning(f"TimescaleDB unavailable ({e}). Writes disabled.")
            return None

    def _ensure_table(self) -> bool:
        conn = self._connect()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Table creation failed: {e}")
            return False
        # Hypertable conversion is best-effort — requires TimescaleDB extension.
        # Plain PostgreSQL tables work fine without it.
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_HYPERTABLE_SQL)
            conn.commit()
        except Exception:
            conn.rollback()
            logger.info(f"TimescaleDB not available — {_TABLE} will use plain PostgreSQL.")
        return True

    def write(self, df: pd.DataFrame) -> int:
        """Insert aggregated rows. Returns count of rows written, 0 on any failure."""
        if df is None or df.empty:
            return 0
        conn = self._connect()
        if not conn:
            return 0
        self._ensure_table()

        present = [c for c in _INSERT_COLS if c in df.columns]
        db_cols = ["time" if c == "timestamp" else c for c in present]
        placeholders = ",".join(["%s"] * len(db_cols))
        sql = (
            f"INSERT INTO {_TABLE} ({','.join(db_cols)}) "
            f"VALUES ({placeholders}) ON CONFLICT DO NOTHING;"
        )
        rows = [tuple(row.get(c) for c in present) for _, row in df.iterrows()]
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows)
            conn.commit()
            logger.info(f"Wrote {len(rows)} rows to TimescaleDB.")
            return len(rows)
        except Exception as e:
            conn.rollback()
            logger.warning(f"TimescaleDB write failed: {e}")
            return 0

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
