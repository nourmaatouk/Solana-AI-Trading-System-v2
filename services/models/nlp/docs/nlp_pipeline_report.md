# NLP Pipeline Report

## Purpose

The NLP module transforms unstructured crypto news into structured 15-minute sentiment features for the trading classifier.

The NLP model does not execute trades and does not decide market direction by itself. It produces context features that help the XGBoost classifier understand whether recent news pressure is positive, negative, neutral, high-volume, or event-driven.

## Current Architecture

```text
Layer 1: News Sources
  - GDELT Doc API
  - RSS feeds

Layer 2: NLP Processing
  - deduplicate articles
  - clean text
  - relevance filter
  - FinBERT sentiment inference
  - event type detection
  - event impact weighting

Layer 3: Feature Aggregation
  - group into 15-minute buckets
  - compute sentiment ratios
  - compute event features
  - compute momentum/spike features

Layer 4: Classification Bridge
  - write latest sentiment CSV
  - merge with market/orderbook features
  - feed XGBoost classifier
```

## Data Sources

`src/fetchers/gdelt_fetcher.py`

Uses GDELT to search for configured Solana keywords:

```text
Solana
SOL token
Solana blockchain
```

The setting `GDELT_FETCH_INTERVAL_MINUTES` controls the query lookback window. It does not schedule the script.

`src/fetchers/rss_fetcher.py`

Reads configured crypto RSS feeds and normalizes each article into a shared dictionary containing source, title, URL, description, and timestamp.

Default feeds:

```text
CoinDesk
CryptoSlate
Cointelegraph
```

## Article Processing

`src/processing/text_cleaner.py` prepares raw article data before inference.

It performs:

```text
HTML removal
URL removal
symbol cleanup
lowercasing
space normalization
minimum-token filtering
keyword relevance scoring
event type classification
event weight assignment
```

Relevance scoring prefers Solana-specific and crypto-market terms such as:

```text
solana
sol
crypto market
etf
hack
exploit
outage
whale
liquidity
regulation
stablecoin
dex
defi
sec
bullish
bearish
airdrop
listing
```

## Sentiment Model

`src/models/finbert_wrapper.py` uses:

```text
ProsusAI/finbert
```

The model returns three probabilities:

```text
positive_prob
negative_prob
neutral_prob
```

The trading feature score is:

```text
sentiment_score = positive_prob - negative_prob
```

Interpretation:

```text
near +1.0: strongly positive
near  0.0: neutral or mixed
near -1.0: strongly negative
```

## Event Logic

The NLP pipeline classifies articles into event categories:

```text
hack/security
outage/network
whale
ETF/regulation
partnership
market_move
general
```

Event weights:

```text
hack/security: 3.0
outage/network: 3.0
whale: 2.0
partnership: 2.0
ETF/regulation: 1.0
market_move: 1.0
general: 0.5
```

The weighted signal is:

```text
event_signal = sentiment_score * event_weight
```

This gives higher influence to articles that are more likely to move the market, such as hacks, outages, large transfers, or major partnerships.

## Aggregation

`src/aggregation/feature_aggregator.py` groups article rows into 15-minute buckets.

The aggregation creates:

```text
sentiment_mean
sentiment_std
news_volume
positive_ratio
negative_ratio
neutral_ratio
strong_positive_ratio
strong_negative_ratio
max_confidence
event_count
negative_event_count
positive_event_count
high_impact_event_count
high_impact_event
relevance_weighted_sentiment
event_signal_mean
event_signal_sum
gdelt_volume
rss_volume
gdelt_sentiment
rss_sentiment
sentiment_momentum
news_volume_change
sentiment_spike
```

These features are aligned to the same 15-minute timeframe as the market candles.

## Runtime Modes

### Offline Mode

Command:

```bash
cd services/models/nlp
python src/main.py
```

Outputs:

```text
services/models/nlp/data/processed/processed_articles.csv
services/models/nlp/data/processed/sentiment_features_15m.csv
```

Use this mode to inspect article-level sentiment quality or build historical sentiment datasets.

### Realtime Bridge Mode

Command from the repository root:

```bash
python services/models/classification/realtime_sentiment_fetcher.py
```

Outputs:

```text
services/models/classification/data/realtime/latest_sentiment_15m.csv
services/models/classification/data/realtime/processed_urls.txt
```

This script is in the classification folder because it writes directly into the classifier realtime feature directory, but it imports and uses the NLP code from `services/models/nlp/src`.

### HTML Dashboard Mode

Command from the repository root:

```bash
python services/models/nlp/22_nlp_sentiment_dashboard.py
```

Output:

```text
services/models/nlp/reports/nlp_sentiment_dashboard.html
```

The dashboard is a static HTML file. Open it in a browser to inspect:

```text
latest sentiment_mean
latest news_volume
positive and negative ratios
high-impact event count
sentiment trend chart
news volume chart
event distribution
recent sentiment buckets
top article-level event signals
input file status
```

The dashboard can read all three available result files:

```text
services/models/nlp/data/processed/sentiment_features_15m.csv
services/models/nlp/data/processed/processed_articles.csv
services/models/classification/data/realtime/latest_sentiment_15m.csv
```

If the offline NLP CSVs are missing, it still shows the realtime sentiment bridge output when available. If all inputs are missing, it generates a safe empty-state page.

## Automatic Reruns

The NLP code does not rerun automatically by itself.

Each of these commands runs once, then exits:

```bash
python services/models/nlp/src/main.py
python services/models/classification/realtime_sentiment_fetcher.py
```

To make it automatic, use cron, systemd, Docker, Airflow, Prefect, or another scheduler.

Recommended live schedule:

```text
Every 5 minutes:
  python services/models/classification/realtime_sentiment_fetcher.py

Every 15 minutes after candle close:
  python services/models/classification/run_realtime_pipeline.py
```

Why 5 minutes for NLP:

```text
GDELT_FETCH_INTERVAL_MINUTES=5 means the fetcher asks GDELT for articles seen in the last 5 minutes.
Running the fetcher every 5 minutes keeps the lookback and scheduler aligned.
The URL cache prevents already processed articles from being reused.
```

For more robust production behavior, use a 10 or 15 minute GDELT lookback with URL deduplication. This creates overlap and reduces the risk of missing delayed articles.

## No-News Behavior

Live trading must not break when there is no fresh news.

The realtime sentiment bridge handles this by:

```text
1. loading the previous sentiment row when available
2. decaying sentiment values by 50 percent
3. resetting news volume and event counts to zero
4. writing a valid latest_sentiment_15m.csv
```

If there is no previous row, it writes a neutral zero row.

This keeps the classifier feature contract stable.

## How The Classifier Uses NLP

The classifier does not directly read raw articles.

It reads 15-minute numeric sentiment features after `realtime_merge_features.py` combines:

```text
market features
technical indicators
orderbook features
sentiment features
```

NLP can influence the classifier through features such as:

```text
sentiment_mean
news_volume
negative_ratio
positive_ratio
event_signal_sum
high_impact_event
sentiment_spike
```

The XGBoost model then decides whether the market is more likely to move up or down over its trained horizon.

## Testing

Run the NLP test suite:

```bash
python -m pytest services/models/nlp/tests
```

The tests cover:

```text
GDELT fetcher behavior
RSS parsing behavior
text cleaning
relevance filtering
FinBERT wrapper output shape
sentiment pipeline schema
event detection
15-minute aggregation
CSV writing
market/sentiment merge
Layer 2 schema contracts
```

Most tests use mocks so they do not need to download FinBERT or call live news APIs.

## Known Limitations

GDELT and RSS are not perfect realtime sources. News may arrive late.

FinBERT is trained for finance language, not specifically crypto-native slang.

Short headlines can be noisy. The relevance filter reduces noise but will not catch every irrelevant article.

CSV storage is acceptable for local testing but not ideal for production. TimescaleDB or another append-safe store is better for long-running services.

The realtime bridge keeps a simple URL cache. For production, use a database table or Redis set with retention.

## Recommended Next Improvements

Add a scheduler service for the realtime sentiment bridge.

Use overlapping GDELT lookbacks with durable deduplication.

Store article-level NLP events in TimescaleDB.

Expose latest sentiment features through the backend API.

Connect the static NLP dashboard to the backend API so the React dashboard can show recent articles, high-impact events, and sentiment spikes.

Calibrate whether sentiment features improve classification out-of-sample before letting them affect execution size.
