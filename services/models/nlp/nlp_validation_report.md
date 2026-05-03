# NLP Pipeline Validation Report
Generated on: 2026-04-30 14:19:24.997617

## 📊 Dataset Overview
- **Total Articles Analyzed**: 32
- **Sources**: <ArrowStringArray>
['rss']
Length: 1, dtype: str

## 🎯 Top 10 High-Signal Articles (Event Weighted)
*Prioritizing items with high event_weight and directional sentiment.*

| title                                                                                                | event_type     |   event_weight |   sentiment_score |   event_signal |
|:-----------------------------------------------------------------------------------------------------|:---------------|---------------:|------------------:|---------------:|
| Should users be allowed to bet on war and death in prediction markets?                               | hack/security  |              3 |         -0.923022 |      -2.76907  |
| Crypto hack losses top $630M in April, highest since February 2025                                   | hack/security  |              3 |         -0.805241 |      -2.41572  |
| AI-driven hacks could kill DeFi — unless projects act now                                            | hack/security  |              3 |         -0.622583 |      -1.86775  |
| South Korea’s Shinhan Card taps Solana to test real-world stablecoin payments                        | partnership    |              2 |          0.799595 |       1.59919  |
| XRP hints at 30% spike, Bitcoin ETFs post 9-day inflow streak: Hodler’s Digest, April 19 – 25        | ETF/regulation |              1 |          0.906847 |       0.906847 |
| US, UAE and China joint effort dismantles 9 crypto scam centers                                      | ETF/regulation |              1 |         -0.882085 |      -0.882085 |
| Bitcoin falls as traders cut risk ahead of FOMC: Will TradFi, spot ETF volumes bolster $70K support? | ETF/regulation |              1 |         -0.874325 |      -0.874325 |
| Bitcoin futures signal caution as long-to-short ratio signals positioning shift                      | whale          |              2 |         -0.376289 |      -0.752578 |
| US seized $500M in Iranian crypto assets, Treasury secretary says                                    | ETF/regulation |              1 |         -0.684872 |      -0.684872 |
| Bitcoin ‘on track’ for $90K, ETFs pull in nearly $1B: Hodler’s Digest, April 12 – 18                 | ETF/regulation |              1 |          0.657987 |       0.657987 |

## 🔍 Event Analysis
### Article Counts by Category
| event_type     |   count |
|:---------------|--------:|
| ETF/regulation |      16 |
| general        |      11 |
| hack/security  |       3 |
| partnership    |       1 |
| whale          |       1 |

![Weight Distribution](reports/images/weight_distribution.png)

- **High Impact (Weight >= 2.0) Events Detected**: 5

## ⚖️ Sentiment Profile
![Sentiment Distribution](reports/images/sentiment_distribution.png)

### Sentiment Breakdown
| sentiment_label   |   count |
|:------------------|--------:|
| neutral           |      17 |
| negative          |       8 |
| positive          |       7 |