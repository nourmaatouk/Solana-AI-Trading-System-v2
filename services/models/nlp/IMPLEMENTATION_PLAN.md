# NLP/Sentiment Analysis Module - Implementation Plan

## 1. DATA SOURCES SELECTION

### Primary Sources

#### **GDELT (Global Database of Events, Language and Tone)**
- **Why**: Comprehensive news coverage, free tier with REST API, includes tone/emotion metrics
- **Coverage**: 200+ countries, global news feeds in 100+ languages
- **Advantages**: 
  - Historical data available (1979-present)
  - Real-time updates (15-min refresh)
  - Pre-computed analysis (tone, intensity, relevance)
  - No authentication required for basic queries
- **Rate limits**: 15,000 requests/15min from single IP
- **Cost**: Free tier sufficient; premium available for higher volume

#### **RSS Feeds (Crypto-Specific News)**
- **Sources to ingest**:
  - CoinDesk (`https://www.coindesk.com/arc/outboundfeeds/rss/`)
  - CryptoSlate (`https://cryptoslate.com/feed/`)
  - CoinTelegraph (`https://cointelegraph.com/feed/`)
  - Solana-specific: Solana official blog
- **Why**: 
  - Crypto-domain expertise (not generic market news)
  - Fast publication (minutes, not hours)
  - Structured metadata (title, description, published date)
  - Reliable feeds with consistent formatting
  - Lower latency than GDELT
- **Advantages**: 
  - Free, no authentication needed (except CoinDesk may require API key)
  - Easy to parse with standard libraries (feedparser)
  - Token-specific news (more relevant than macro news)
- **Trade-off**: Smaller volume than GDELT, but higher relevance

#### **Reddit (Optional Secondary)**
- **Subreddits**: r/solana, r/SolanaDev, r/crypto, r/defi
- **Why**: Real-time community sentiment, high volume during market moves
- **Advantages**: 
  - Grassroots sentiment (retail traders)
  - Comment volume as interest proxy
  - Trending topics identified early
- **Challenges**:
  - PRAW API rate-limited (60 requests/min unauthenticated)
  - High noise:signal ratio
  - Requires authentication for better rates
  - Secondary priority; use if infrastructure supports it

---

## 2. MODULE OBJECTIVE

**Convert market news and social signals into quantified sentiment features for ML model input.**

Specifically:
- Extract actionable sentiment from diverse text sources (news, blogs, social)
- Produce time-bucketed features aligned to trading timeframe (15-min)
- Enable early detection of sentiment shifts that precede price movements
- Provide interpretable sentiment scores for feature attribution (SHAP analysis)
- Feed into fusion model as auxiliary signal alongside price/microstructure

---

## 3. INPUT DATA

### Data Sources Format

#### From GDELT API
```
{
  "source": "gdelt",
  "id": "unique_event_id",
  "title": "Article headline",
  "text": "Article body (full or summary)",
  "url": "https://...",
  "published_at": "2026-04-29T14:30:00Z",
  "iso_language": "en",
  "tone": {
    "avg": 2.5,           // GDELT pre-computed tone
    "positive": 0.6,
    "negative": 0.1
  }
}
```

#### From RSS Feeds
```
{
  "source": "rss_coindesk",
  "feed_source": "CoinDesk",
  "title": "Solana Network Activity Reaches 6-Month High",
  "description": "Network utilization...",
  "text": "Full article text extracted from content",
  "url": "https://coindesk.com/...",
  "published_at": "2026-04-29T13:45:00Z",
  "language": "en"
}
```

#### From Reddit (Optional)
```
{
  "source": "reddit",
  "subreddit": "solana",
  "title": "[Discussion] SOL trading signals",
  "text": "Post body + top comments aggregated",
  "url": "https://reddit.com/r/solana/...",
  "published_at": "2026-04-29T12:30:00Z",
  "upvotes": 245,
  "comment_count": 87
}
```

### Data Collection Format
- **Storage**: JSON Lines (.jsonl) for raw articles
- **Queue**: Redis Streams for ingest pipeline
- **Batch size**: 100 articles per batch
- **Refresh interval**: 5 minutes for new articles

---

## 4. PROCESSING STEPS

### Step 1: Fetch Data
**Purpose**: Retrieve new articles from all sources continuously

**Sources**:
- GDELT: Query API for articles mentioning `SOL`, `Solana`, `token trading`
- RSS: Poll feeds every 5 minutes
- Reddit: Query new posts from subscribed subreddits

**Pseudocode**:
```
while True:
    # Fetch from each source
    gdelt_articles = query_gdelt_api(keywords=["SOL", "Solana"])
    rss_articles = fetch_rss_feeds(feeds=RSS_FEED_LIST)
    reddit_articles = fetch_reddit_posts(subreddits=SUBREDDITS)
    
    # Merge and deduplicate
    all_articles = deduplicate_by_url(gdelt_articles + rss_articles + reddit_articles)
    
    # Push to Redis stream
    for article in all_articles:
        redis_stream.add("nlp:raw_articles", article)
    
    sleep(5 minutes)
```

---

### Step 2: Clean Text
**Purpose**: Normalize raw text for model input

**Operations**:
- Remove HTML tags (`<p>`, `<div>`, etc.)
- Remove URLs and email addresses
- Remove emoji and special Unicode characters
- Normalize whitespace (multiple spaces → single space)
- Convert to lowercase for consistency
- Strip leading/trailing whitespace
- Handle encoding issues (UTF-8)

**Constraints**:
- Keep text under 512 tokens (FinBERT max sequence length)
- Truncate long articles (keep first 512 tokens)
- Skip articles with <20 tokens (too short for sentiment)

**Pseudocode**:
```
def clean_text(raw_text):
    # Remove HTML
    text = remove_html_tags(raw_text)
    
    # Remove URLs
    text = regex_sub(r'https?://\S+', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove special chars
    text = regex_sub(r'[^\w\s\.\!\?\-]', '', text)
    
    # Truncate if needed
    tokens = tokenizer.encode(text)
    if len(tokens) > 512:
        tokens = tokens[:512]
        text = tokenizer.decode(tokens)
    
    return text.strip()
```

---

### Step 3: Apply FinBERT Model
**Purpose**: Convert cleaned text to sentiment scores

**Model**:
- FinBERT (huggingface: `ProsusAI/finbert`)
- Pre-trained on financial text (SEC filings, earnings calls)
- Output: [negative, neutral, positive] logits
- Inference: Single forward pass per article

**Pseudocode**:
```
def compute_sentiment(cleaned_text):
    # Tokenize
    inputs = tokenizer.encode_plus(
        cleaned_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    
    # Inference
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    
    # Extract logits
    logits = outputs[0].detach().cpu().numpy()[0]
    
    # Softmax to probabilities
    probs = softmax(logits)  # [neg_prob, neu_prob, pos_prob]
    
    # Compute sentiment score: range [-1, +1]
    sentiment = pos_prob - neg_prob
    
    return {
        "sentiment_score": sentiment,      # -1 (very negative) to +1 (very positive)
        "negative_prob": probs[0],
        "neutral_prob": probs[1],
        "positive_prob": probs[2],
        "confidence": max(probs)           # max probability as confidence
    }
```

---

### Step 4: Compute Sentiment Score
**Purpose**: Convert FinBERT output to interpretable scores

**Score Format**:
- **Raw sentiment**: Range [-1, +1]
  - -1: very negative
  - 0: neutral
  - +1: very positive
- **Weighted sentiment**: Multiply by confidence score
- **Direction**: binary classification (bullish/bearish)

---

### Step 5: Group into 15-Minute Intervals
**Purpose**: Align sentiment features to trading timeframe (15-min candles)

**Bucketing Logic**:
```
def round_to_15min(timestamp):
    # Round down to nearest 15-min boundary
    rounded = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
    return rounded
```

**Aggregation per 15-min bucket**:
- Collect all articles with `published_at` in window [T, T+15min)
- Compute statistics across articles

---

### Step 6: Save Features
**Purpose**: Persist computed features to queryable storage

**Outputs**:
1. **TimescaleDB** (historical + materialized views)
   - Insert feature rows (timestamp, sentiment metrics, source_breakdown)
   - Create materialized view for quick queries
   
2. **Redis Cache** (real-time)
   - Current 15-min window in-flight data
   - Quick lookup for dashboard/API

**Pseudocode**:
```
def save_features(bucket_timestamp, articles_in_bucket):
    # Aggregate statistics
    sentiments = [article.sentiment_score for article in articles_in_bucket]
    
    feature_row = {
        "timestamp": bucket_timestamp,
        "sentiment_mean": mean(sentiments),
        "sentiment_std": std(sentiments),
        "positive_ratio": count(s > 0.1) / len(sentiments),
        "negative_ratio": count(s < -0.1) / len(sentiments),
        "news_volume": len(sentiments),
        "sentiment_spike": detect_spike(sentiments),
        "gdelt_volume": count where source="gdelt",
        "rss_volume": count where source="rss",
        "reddit_volume": count where source="reddit",
        "gdelt_sentiment": mean(sentiments where source="gdelt"),
        "rss_sentiment": mean(sentiments where source="rss"),
        "reddit_sentiment": mean(sentiments where source="reddit"),
    }
    
    # Insert to TimescaleDB
    timescale.insert("nlp_features_15m", feature_row)
    
    # Update Redis cache
    redis.set(f"nlp:current_bucket:{bucket_timestamp}", json(feature_row))
```

---

## 5. OUTPUT FEATURES

### Feature Schema (15-minute intervals)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `timestamp` | int | Unix epoch | Window start time (rounded to 15-min) |
| `sentiment_mean` | float | [-1, +1] | Average sentiment across all articles |
| `sentiment_std` | float | [0, 1] | Standard deviation (dispersion) |
| `positive_ratio` | float | [0, 1] | Fraction of articles with sentiment > 0.1 |
| `negative_ratio` | float | [0, 1] | Fraction of articles with sentiment < -0.1 |
| `neutral_ratio` | float | [0, 1] | Fraction with sentiment in [-0.1, +0.1] |
| `news_volume` | int | ≥0 | Total count of articles |
| `sentiment_spike` | float | ≥0 | Magnitude of sentiment change vs 1-hour rolling avg |
| `gdelt_sentiment` | float | [-1, +1] | Source-specific sentiment (GDELT only) |
| `rss_sentiment` | float | [-1, +1] | Source-specific sentiment (RSS only) |
| `reddit_sentiment` | float | [-1, +1] | Source-specific sentiment (Reddit only) |
| `gdelt_volume` | int | ≥0 | Article count from GDELT |
| `rss_volume` | int | ≥0 | Article count from RSS feeds |
| `reddit_volume` | int | ≥0 | Post count from Reddit |
| `max_confidence` | float | [0, 1] | Max FinBERT confidence in bucket |

### Example Output Row
```json
{
  "timestamp": 1746028800,
  "sentiment_mean": 0.35,
  "sentiment_std": 0.22,
  "positive_ratio": 0.65,
  "negative_ratio": 0.10,
  "neutral_ratio": 0.25,
  "news_volume": 20,
  "sentiment_spike": 0.42,
  "gdelt_sentiment": 0.28,
  "rss_sentiment": 0.45,
  "reddit_sentiment": 0.32,
  "gdelt_volume": 8,
  "rss_volume": 9,
  "reddit_volume": 3,
  "max_confidence": 0.89
}
```

---

## 6. TESTS

### Unit Tests

#### Test 1: Data Fetcher
- **test_gdelt_api_connectivity**: Verify API returns non-empty results
- **test_rss_feed_parsing**: Parse feed, verify article count > 0
- **test_reddit_authentication**: Connect with valid credentials
- **test_deduplication**: Same URL not inserted twice

#### Test 2: Text Cleaning
- **test_html_removal**: `"<p>hello</p>"` → `"hello"`
- **test_url_removal**: URLs stripped from text
- **test_emoji_removal**: Emoji characters removed
- **test_truncation**: Long text truncated to 512 tokens
- **test_short_text_rejection**: Text < 20 tokens skipped
- **test_whitespace_normalization**: Multiple spaces normalized

#### Test 3: FinBERT Model
- **test_sentiment_range**: Output in [-1, +1]
- **test_probabilities_sum_to_one**: neg + neutral + positive ≈ 1.0
- **test_known_positive_text**: "Great news" → sentiment > 0.5
- **test_known_negative_text**: "Crash expected" → sentiment < -0.5
- **test_model_consistency**: Same text gives same sentiment (deterministic)
- **test_model_batching**: Batch inference vs single match results

#### Test 4: Aggregation
- **test_empty_bucket**: No articles → features = null or zeros
- **test_single_article**: Single article → mean = that article's sentiment
- **test_statistics**: Verify mean/std computation with known data
- **test_spike_detection**: Spike flag when sentiment changes > threshold
- **test_ratio_calculation**: Ratios sum close to 1.0

#### Test 5: Feature Storage
- **test_timescale_insertion**: Row inserted successfully
- **test_redis_cache_update**: Cache key updated with feature row
- **test_timestamp_bucketing**: Articles grouped into correct 15-min bucket
- **test_feature_schema**: All required columns present and correct type

### Integration Tests
- **test_full_pipeline**: Fetch → Clean → Sentiment → Aggregate → Store
- **test_concurrent_articles**: Multiple articles in same 15-min window handled
- **test_duplicate_handling**: Same article from multiple sources counted once
- **test_feature_linearity**: Double articles → features scale appropriately

### Validation Tests
- **test_sentiment_vs_price**: (Optional) Verify sentiment spikes correlate with price moves (ground truth)
- **test_source_correlation**: Check correlation between sources
- **test_temporal_consistency**: Features monotonic within buckets

---

## 7. FOLDER STRUCTURE

```
services/models/nlp/
│
├── README.md                          # Module overview
├── IMPLEMENTATION_PLAN.md             # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── Dockerfile                         # Container definition
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # Orchestrator/entrypoint
│   ├── config.py                      # Configuration loader
│   │
│   ├── fetchers/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base class
│   │   ├── gdelt_fetcher.py           # GDELT API client
│   │   ├── rss_fetcher.py             # RSS feed parser
│   │   └── reddit_fetcher.py          # (Optional) Reddit client
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py            # HTML, URL, emoji removal
│   │   └── sentiment_pipeline.py      # Orchestrates full processing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── finbert_wrapper.py         # FinBERT inference
│   │   ├── model_config.py            # Model hyperparameters
│   │   └── constants.py               # Sentiment thresholds, etc.
│   │
│   ├── aggregation/
│   │   ├── __init__.py
│   │   └── feature_aggregator.py      # 15-min bucketing & stats
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── timescale_writer.py        # TimescaleDB insert
│   │   └── redis_cache.py             # Redis cache manager
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       ├── constants.py               # Global constants (thresholds, URLs)
│       └── deduplicator.py            # Handle duplicate articles
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_fetchers.py               # Data fetcher tests
│   ├── test_text_cleaning.py          # Text cleaning tests
│   ├── test_finbert.py                # Model inference tests
│   ├── test_aggregation.py            # Feature aggregation tests
│   ├── test_storage.py                # Database/cache tests
│   ├── test_integration.py            # End-to-end pipeline tests
│   │
│   └── fixtures/
│       ├── sample_articles.json       # Mock article data
│       ├── gdelt_response.json        # Mock GDELT API response
│       ├── rss_feed.xml               # Mock RSS feed
│       └── test_data.py               # Test utilities
│
├── data/
│   ├── raw/                           # Raw articles (ingest logs)
│   ├── processed/                     # Cleaned + sentiment
│   └── logs/                          # Pipeline execution logs
│
└── docker-compose.yml                 # Local dev environment
```

---

## 8. DEPENDENCIES & CONSTRAINTS

### Python Dependencies
```
transformers>=4.30          # FinBERT model
torch>=2.0                  # PyTorch for inference
feedparser>=6.0             # RSS feed parsing
requests>=2.28              # HTTP client
praw>=7.7                   # Reddit API (optional)
psycopg2-binary>=2.9        # TimescaleDB driver
redis>=4.5                  # Redis client
numpy>=1.24                 # Numerical operations
pandas>=1.5                 # Data manipulation
python-dotenv>=0.21         # Environment config
```

### Constraints
- **FinBERT inference latency**: ~100-200ms per article (GPU accelerated)
- **API rate limits**:
  - GDELT: 15,000 requests per 15 min
  - RSS: Typically no limits (polling friendly)
  - Reddit: 60 requests/min (unauthenticated), unlimited with premium
- **Storage**: ~1KB per feature row; expect ~1GB/month at current volume
- **Memory**: ~2GB for FinBERT model + batch processing

---

## 9. TIMELINE & DEPENDENCIES

### Phase 1 (Week 1): Infrastructure & Fetchers
- Set up project structure
- Implement base fetcher class
- Implement GDELT fetcher + tests
- Implement RSS fetcher + tests

### Phase 2 (Week 2): Processing
- Implement text cleaning
- Integrate FinBERT model
- Unit tests for cleaning & model

### Phase 3 (Week 3): Aggregation & Storage
- Implement 15-min bucketing
- TimescaleDB writer
- Redis cache manager
- Integration tests

### Phase 4 (Week 4): Optimization & Deployment
- Performance profiling
- Add Reddit fetcher (optional)
- Docker containerization
- Production deployment

---

## 10. INTEGRATION WITH MAIN SYSTEM

### Inputs to Fusion Model
```
services/models/fusion/
├── features/
│   ├── price_features.py        # Price/microstructure
│   ├── nlp_features.py          # ← NLP output features
│   ├── gnn_features.py          # Market graph
│   └── fusion_aggregator.py     # Combine all features
```

### Feature Materialization in TimescaleDB
```sql
CREATE TABLE nlp_features_15m (
    timestamp BIGINT NOT NULL,
    sentiment_mean FLOAT,
    sentiment_std FLOAT,
    positive_ratio FLOAT,
    negative_ratio FLOAT,
    neutral_ratio FLOAT,
    news_volume INT,
    sentiment_spike FLOAT,
    gdelt_sentiment FLOAT,
    rss_sentiment FLOAT,
    reddit_sentiment FLOAT,
    gdelt_volume INT,
    rss_volume INT,
    reddit_volume INT,
    max_confidence FLOAT,
    PRIMARY KEY (timestamp)
) WITH (timescaledb.compress_segmentby = 'timestamp');
```

### Redis Cache Keys
```
nlp:current_bucket:{timestamp}          # Current 15-min aggregated features
nlp:raw_articles                        # Redis Stream of raw articles
nlp:processed_articles                  # Redis Stream of scored articles
```

---

## Summary

This plan provides a **simple, practical, and scalable** sentiment analysis module that:
- ✅ Uses reliable, free data sources (GDELT + RSS)
- ✅ Leverages domain-specific FinBERT model
- ✅ Produces time-aligned features for 15-min trading
- ✅ Integrates seamlessly with existing fusion model
- ✅ Includes comprehensive testing strategy
- ✅ Follows modular, maintainable architecture
