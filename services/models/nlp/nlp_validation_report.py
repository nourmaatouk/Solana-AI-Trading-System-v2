import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_nlp_report():
    print("Generating NLP Validation Report...")
    
    # Paths
    NLP_DIR = Path(__file__).resolve().parent
    DATA_DIR = NLP_DIR / "data" / "processed"
    ARTICLE_PATH = DATA_DIR / "processed_articles.csv"
    REPORT_PATH = NLP_DIR / "nlp_validation_report.md"
    IMAGE_DIR = NLP_DIR / "reports" / "images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    overview_md = """# NLP Validation Report

This report summarizes the current NLP sentiment pipeline and the latest available validation snapshot.

For the full architecture and runtime explanation, see:

```text
services/models/nlp/docs/nlp_pipeline_report.md
```

## What The NLP Pipeline Does

The NLP module fetches crypto/Solana news, cleans and filters article text, scores it with FinBERT, detects event type, applies event impact weights, and aggregates everything into 15-minute sentiment features.

It does not place trades and it does not decide UP/DOWN alone. The classification model consumes the aggregated sentiment features together with market, technical, and orderbook features.

## Automatic Rerun Behavior

The NLP scripts are one-shot scripts.

They run once and exit:

```bash
python services/models/nlp/src/main.py
python services/models/classification/realtime_sentiment_fetcher.py
```

`GDELT_FETCH_INTERVAL_MINUTES=5` means "fetch articles seen in the last 5 minutes." It does not mean "run this script automatically every 5 minutes."

Recommended scheduler:

```text
NLP sentiment fetcher: every 5 minutes
Classification pipeline: every 15 minutes after candle close
```

## Runtime Outputs

Offline NLP pipeline:

```text
services/models/nlp/data/processed/processed_articles.csv
services/models/nlp/data/processed/sentiment_features_15m.csv
```

Realtime classifier bridge:

```text
services/models/classification/data/realtime/latest_sentiment_15m.csv
services/models/classification/data/realtime/processed_urls.txt
```

Static HTML dashboard:

```bash
python services/models/nlp/22_nlp_sentiment_dashboard.py
```

```text
services/models/nlp/reports/nlp_sentiment_dashboard.html
```

Open the generated HTML file in a browser to inspect latest sentiment, bucket trends, event distribution, and top article signals.
"""

    # 1. Load Data
    if not ARTICLE_PATH.exists():
        print(f"CRITICAL WARNING: {ARTICLE_PATH} not found.")
        print("Please run the NLP pipeline (src/main.py) or backfiller to generate article-level sentiment data.")
        
        # Save a placeholder report warning the user
        with open(REPORT_PATH, "w") as f:
            f.write(overview_md)
            f.write("\n## Validation Snapshot\n\n")
            f.write("Status: DATA MISSING\n\n")
            f.write(f"The file `processed_articles.csv` was not found at {ARTICLE_PATH}. ")
            f.write("Run the pipeline first to generate data.\n")
        return

    df_articles = pd.read_csv(ARTICLE_PATH)
    
    # 2. Distribution: Event Weight
    plt.figure(figsize=(10, 6))
    if "event_weight" in df_articles.columns:
        sns.histplot(df_articles["event_weight"], bins=10, color="green")
        plt.title("NLP Distribution: Event Weights")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "weight_distribution.png")
        plt.close()

    # 3. Distribution: Sentiment Score
    plt.figure(figsize=(10, 6))
    if "sentiment_score" in df_articles.columns:
        sns.histplot(df_articles["sentiment_score"], bins=30, kde=True, color="blue")
        plt.title("NLP Distribution: Sentiment Scores")
        plt.savefig(IMAGE_DIR / "sentiment_distribution.png")
        plt.close()

    # 4. Generate Markdown
    md_content = f"""{overview_md}

## Validation Snapshot

Generated on: {pd.Timestamp.now()}

## Dataset Overview
- **Total Articles Analyzed**: {len(df_articles)}
- **Sources**: {df_articles['source'].unique() if 'source' in df_articles.columns else 'Unknown'}

## Top 10 High-Signal Articles (Event Weighted)
*Prioritizing items with high event_weight and directional sentiment.*

"""
    if "event_signal" in df_articles.columns:
        # Absolute signal identifies both high-impact bullish and bearish news
        top_10 = df_articles.reindex(df_articles["event_signal"].abs().sort_values(ascending=False).index).head(10)
        cols = ["title", "event_type", "event_weight", "sentiment_score", "event_signal"]
        md_content += top_10[cols].to_markdown(index=False)
    else:
        md_content += "Event signal column missing.\n"

    md_content += "\n\n## Event Analysis\n"
    if "event_type" in df_articles.columns:
        counts = df_articles["event_type"].value_counts()
        md_content += "### Article Counts by Category\n"
        md_content += counts.to_markdown()
        md_content += "\n\n![Weight Distribution](reports/images/weight_distribution.png)\n"
        
        hi_count = 0
        if "high_impact_event" in df_articles.columns:
            hi_count = df_articles["high_impact_event"].sum()
        else:
            hi_count = df_articles[df_articles["event_type"].isin(["hack/security", "outage/network"])].shape[0]
            
        md_content += f"\n- **High Impact (Weight >= 2.0) Events Detected**: {hi_count}\n"

    md_content += "\n## Sentiment Profile\n"
    md_content += "![Sentiment Distribution](reports/images/sentiment_distribution.png)\n"
    
    if "sentiment_label" in df_articles.columns:
        sent_counts = df_articles["sentiment_label"].value_counts()
        md_content += "\n### Sentiment Breakdown\n"
        md_content += sent_counts.to_markdown()

    # Save Report
    with open(REPORT_PATH, "w") as f:
        f.write(md_content)
        
    print(f"Validation Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    generate_nlp_report()
