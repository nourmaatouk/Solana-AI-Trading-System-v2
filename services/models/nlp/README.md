# NLP Sentiment Module

This module converts crypto and Solana-related news into numeric sentiment features that the market classification model can use.

It does not predict UP or DOWN by itself. Its job is to fetch articles from GDELT, RSS feeds, and Reddit, score them with FinBERT, aggregate the result into 15-minute buckets, and export features such as `sentiment_mean`, `news_volume`, `event_signal_sum`, and `sentiment_spike` to CSV, Redis, and TimescaleDB.

## Where It Fits

```text
GDELT + RSS + Reddit
         |
         v
NLP fetchers (gdelt_fetcher, rss_fetcher, reddit_fetcher)
         |
         v
deduplication (deduplicator)
         |
         v
text cleaning + relevance filter (text_cleaner)
         |
         v
FinBERT sentiment scoring (finbert_wrapper — GPU if available)
         |
         v
event classification + impact weights
         |
         v
15-minute sentiment features (feature_aggregator)
         |
         v
CSV  ──── Redis cache ──── TimescaleDB
         |
         v
classification feature merge
```

The live classifier consumes the NLP output through:

```text
services/models/classification/data/realtime/latest_sentiment_15m.csv
```

## Important Timing Rule

The NLP code is one-shot by default. It runs once each time you execute the script.

`GDELT_FETCH_INTERVAL_MINUTES=5` does not schedule the script every 5 minutes. It only tells GDELT to return articles seen in the last 5 minutes.

For automatic live behavior, use an external scheduler:

```bash
*/5 * * * * cd /path/to/Solana-AI-Trading-System && . .venv/bin/activate && python services/models/nlp/src/main.py
```

Recommended live schedule:

```text
Orderbook collector:          always running
NLP sentiment fetcher:        every 5 minutes
Classification pipeline:      every 15 minutes, after candle close
Dashboard/report refresh:     after classification prediction
```

## Running With Docker

Start the full stack (NLP pipeline + Redis + TimescaleDB):

```bash
cd services/models/nlp
cp .env.example .env        # edit credentials if needed
docker compose up --build
```

To run only the pipeline once against existing Redis/TimescaleDB instances:

```bash
docker compose run --rm nlp-pipeline
```

## Running Locally

Install dependencies:

```bash
pip install -r services/models/nlp/requirements.txt
```

Run the pipeline:

```bash
cd services/models/nlp
python src/main.py
```

This fetches GDELT, RSS, and Reddit articles, deduplicates them, scores them with FinBERT, aggregates to 15-minute buckets, and writes:

```text
data/processed/processed_articles.csv          # article-level, for validation
data/processed/sentiment_features_15m.csv      # aggregated features
Redis:        nlp:sentiment:<timestamp>         # per-bucket cache (TTL = REDIS_TTL_SECONDS)
TimescaleDB:  sentiment_features_15m            # hypertable, one row per 15m bucket
```

### Live Sentiment Bridge

Run from the repository root:

```bash
python services/models/classification/realtime_sentiment_fetcher.py
```

This is the live bridge used by the classification pipeline. It writes:

```text
services/models/classification/data/realtime/latest_sentiment_15m.csv
services/models/classification/data/realtime/processed_urls.txt
```

If no fresh news is found, the pipeline does not crash. It emits a **decayed sentiment row** from the Redis cache (sentiment values halved, `news_volume = 0`) so downstream consumers always receive a non-empty row.

## Module Overview

| Module | Purpose |
|--------|---------|
| `src/fetchers/base.py` | Abstract `BaseFetcher` — all fetchers implement `fetch() -> List[Dict]` |
| `src/fetchers/gdelt_fetcher.py` | GDELT API client with 3-attempt retry and exponential backoff |
| `src/fetchers/rss_fetcher.py` | RSS parser for CoinDesk, CryptoSlate, Cointelegraph |
| `src/fetchers/reddit_fetcher.py` | Reddit public JSON API (no OAuth / no PRAW) |
| `src/utils/deduplicator.py` | Removes duplicate articles by URL, then by title |
| `src/utils/logger.py` | Structured stdout logger — reads `LOG_LEVEL` from config |
| `src/utils/constants.py` | Named constants for thresholds and aggregation settings |
| `src/processing/text_cleaner.py` | HTML/URL/emoji removal, relevance scoring, event classification |
| `src/processing/sentiment_pipeline.py` | Orchestrates cleaning → FinBERT → event weighting |
| `src/processing/merge_sentiment_with_market.py` | Joins historical market OHLCV with sentiment features for training |
| `src/models/finbert_wrapper.py` | ProsusAI/finbert inference — auto-detects CUDA, falls back to CPU |
| `src/models/xgboost_trainer.py` | XGBoost with walk-forward (expanding-window) cross-validation |
| `src/models/constants.py` | FinBERT model name, token limits, walk-forward validation params |
| `src/aggregation/feature_aggregator.py` | Groups articles into 15-min buckets, computes 22+ features |
| `src/storage/csv_writer.py` | Writes article-level and aggregated CSV outputs |
| `src/storage/redis_cache.py` | Caches per-bucket rows in Redis — gracefully disabled if Redis is unavailable |
| `src/storage/timescale_writer.py` | Writes rows to TimescaleDB hypertable — gracefully disabled if DB is unavailable |
| `src/contracts/layer2_contract.py` | Schema normalization — ensures consistent columns across pipeline stages |

## Configuration

All settings are loaded by `src/config.py` (Pydantic-based). Defaults work without a `.env` file. Copy `.env.example` to `.env` to override.

```text
# Core
APP_ENV                       development
LOG_LEVEL                     INFO

# Fetching
RSS_FEEDS                     comma-separated URLs
GDELT_KEYWORDS                Solana,SOL token,Solana blockchain
GDELT_FETCH_INTERVAL_MINUTES  5
REDDIT_SUBREDDITS             solana,CryptoCurrency,CryptoMarkets
REDDIT_FETCH_ENABLED          true

# FinBERT
MAX_TEXT_LENGTH               512
SENTIMENT_THRESHOLD           0.1

# Storage
OUTPUT_DIR                    data/processed
RAW_DIR                       data/raw
TIMEFRAME_MINUTES             15

# Redis
REDIS_URL                     redis://localhost:6379/0
REDIS_TTL_SECONDS             3600

# TimescaleDB
TIMESCALE_DSN                 postgresql://postgres:password@localhost:5432/trading
```

Both `REDIS_URL` and `TIMESCALE_DSN` are optional for local development — the pipeline falls back to CSV-only if either service is unavailable.

## Output Features

15-minute aggregated features written to all three outputs (CSV / Redis / TimescaleDB):

```text
sentiment_mean                 mean FinBERT score across articles in bucket
sentiment_std                  spread of scores
news_volume                    total articles in bucket
positive_ratio                 fraction with score > threshold
negative_ratio                 fraction with score < -threshold
neutral_ratio                  remaining fraction
strong_positive_ratio          fraction with score > 0.8
strong_negative_ratio          fraction with score < -0.8
max_confidence                 highest FinBERT confidence in bucket
event_count                    non-general event articles
negative_event_count
positive_event_count
high_impact_event_count        hack/security or outage/network articles
high_impact_event              1 if any high-impact event present
relevance_weighted_sentiment   sentiment × relevance_score
event_signal_mean
event_signal_sum
gdelt_volume                   articles from GDELT
rss_volume                     articles from RSS
gdelt_sentiment                mean score for GDELT subset
rss_sentiment                  mean score for RSS subset
sentiment_momentum             diff of sentiment_mean (1 step)
news_volume_change             diff of news_volume (1 step)
sentiment_spike                deviation from 4-period rolling mean
```

These columns are normalized by the shared Layer 2 contract so the classifier can safely merge them even when news volume is zero.

## Validation And Reports

Run NLP tests:

```bash
python -m pytest services/models/nlp/tests
```

Generate the HTML dashboard:

```bash
python services/models/nlp/22_nlp_sentiment_dashboard.py
```

Output:

```text
services/models/nlp/reports/nlp_sentiment_dashboard.html
```

Open the generated HTML file in your browser. Do not run the `.html` file as a terminal command.

Generate the validation report after running the NLP pipeline:

```bash
cd services/models/nlp
python nlp_validation_report.py
```

Reports:

```text
services/models/nlp/docs/nlp_pipeline_report.md
services/models/nlp/nlp_validation_report.md
services/models/nlp/reports/nlp_sentiment_dashboard.html
```

## Practical Limits

News is sparse. Many 15-minute windows will have no articles. When that happens the pipeline emits a decayed row from cache instead of an empty output.

GDELT can lag real publication time by several minutes or more.

The first FinBERT run downloads model files from Hugging Face (~500 MB). After that, the local Hugging Face cache makes startup fast.

FinBERT automatically uses CUDA if a GPU is available. CPU mode is slower but fully supported.

The current fetchers are synchronous. For high-volume production use, consider queued ingestion and dedicated inference workers.
