# SOL/USDT Price Direction Prediction — XGBoost ML Pipeline

> **Binary classification pipeline that predicts the 1-hour price direction of Solana (SOL/USDT) using 15-minute OHLCV candlestick data from Binance, a rich set of technical indicators, and rigorous time-series validation.**

---

## Overview

This project builds a **machine learning pipeline** to predict whether the closing price of SOL/USDT will be **higher or lower 1 hour (4 × 15-minute candles) in the future**. The pipeline is intentionally numbered and modular — each script is a self-contained step, making it easy to re-run, debug, or swap any individual component.

The codebase addresses two of the most common mistakes in financial ML:

1. **Data leakage** — a dedicated no-leak preparation step (`07`) splits *before* scaling, so the `StandardScaler` is only fit on training data.
2. **Overly optimistic validation** — a full **walk-forward validation** (`20`) re-trains the model on expanding windows and evaluates on unseen future folds, giving a realistic estimate of live-trading performance.

---

## Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **82.09 %** |
| **Test Precision** | 82.70 % |
| **Test Recall** | 81.30 % |
| **Test F1-Score** | 82.00 % |
| **Test AUC-ROC** | **90.63 %** |
| Best boosting iteration | 272 / 300 |
| Best validation log-loss | 0.3745 |
| Best validation AUC | 91.45 % |
| Training samples | 130,162 |
| Validation samples | 27,892 |
| Test samples | 27,893 |

> The near-balanced test set (13,901 "Down" vs 13,992 "Up") confirms these metrics are meaningful and not inflated by class imbalance.

---

## Project Structure

```
.
├── 01_collect_sol_data_15m.py       # Fetch raw OHLCV from Binance via ccxt
├── 02_build_technical_features.py   # Engineer 30+ technical indicators
├── 03_build_sequences.py            # Clean data, scale, build 3-D sequences
├── 04_split_data.py                 # Chronological 70/15/15 split
├── 05_train_xgboost_baseline.py     # Baseline XGBoost (leaky scaler — reference)
├── 07_prepare_no_leak_data.py       # Leak-free split → scale → sequence
├── 08_train_xgboost_no_leak.py      # XGBoost on no-leak data
├── 20_walk_forward_validation.py    # 5-fold expanding-window walk-forward test
├── 21_xgboost_curves.py             # Full visualisation & JSON report pipeline
│
├── data/
│   └── processed/                   # Intermediate CSVs, .npy arrays, scaler PKLs
│
├── plots/
│   ├── validation_curves/           # Training history plots (log-loss, error, AUC)
│   └── test_curves/                 # Test evaluation plots (ROC, PR, CM, FI…)
│
├── reports/
│   └── model_performance_summary.json   # Full JSON metrics report
│
└── ml_env/                          # (local virtual environment — not committed)
```

---

## Pipeline Steps

Each script is numbered to indicate execution order. Run them sequentially from the project root.

### Step 01 — Data Collection (`01_collect_sol_data_15m.py`)

- Connects to **Binance** via the `ccxt` library.
- Fetches SOL/USDT 15-minute OHLCV candles from **2021-01-01** to present.
- Implements **retry logic** (up to 5 attempts) for network/timeout errors.
- Adds four basic derived columns: `return_1`, `log_return_1`, `high_low_spread`, `open_close_spread`.
- Generates a **binary target**: `target_direction = 1` if `close[t+4] > close[t]` else `0` (1-hour horizon).
- Saves both raw and processed data as CSV + Parquet.

### Step 02 — Technical Feature Engineering (`02_build_technical_features.py`)

Adds 30+ features on top of the raw OHLCV data. See [Feature Engineering](#feature-engineering) for the full list.

### Step 03 — Sequence Building (`03_build_sequences.py`)

- Loads the feature CSV and drops non-feature columns (`timestamp`, `datetime`, `target_direction`).
- Replaces `±inf` and `NaN` with valid values; removes any remaining invalid rows.
- Fits a `StandardScaler` on the **entire dataset** (this is the "leaky" version — used as a baseline reference only).
- Builds **3-D sequences** of shape `(samples, 32, n_features)` suitable for temporal models.
- Saves `X.npy`, `y.npy`, and `scaler.pkl`.

### Step 04 — Data Split (`04_split_data.py`)

- Performs a **chronological** (non-shuffled) 70 / 15 / 15 split on the sequence arrays.
- Saves `X_train.npy`, `X_val.npy`, `X_test.npy` and corresponding `y` arrays.

### Step 05 — Baseline XGBoost (`05_train_xgboost_baseline.py`)

- Flattens 3-D sequences to 2-D for XGBoost.
- Trains `XGBClassifier` (300 estimators, depth 6, lr 0.05) using the leaky-scaled data.
- Reports Accuracy, Precision, Recall, F1, MCC, and AUC-ROC for validation and test sets.
- Saves the model as `.json` and metrics as `.json`.

### Step 07 — No-Leak Data Preparation (`07_prepare_no_leak_data.py`)

- **Fixes the data leakage** from Step 03/04.
- Applies the chronological tabular split **first**, then fits `StandardScaler` **only on the training fold**.
- Transforms validation and test folds using the training scaler only.
- Builds sequences **separately** within each split to avoid boundary contamination.
- Saves `X_*_no_leak.npy` and `scaler_no_leak.pkl`.

### Step 08 — No-Leak XGBoost (`08_train_xgboost_no_leak.py`)

- Same architecture as Step 05 but trained on the correctly prepared no-leak data.
- Produces a fair, unbiased estimate of model performance.

### Step 20 — Walk-Forward Validation (`20_walk_forward_validation.py`)

- Implements **true out-of-sample validation** using an expanding-window scheme over 5 folds.
- Each fold trains on all past data and tests on the immediately following unseen period.
- Computes per-fold and aggregate metrics; compares against the standard train/test split result.
- Interprets the result automatically with four severity bands (genuine edge → no edge).
- Saves fold-level results to `models/walk_forward_results.json`.

### Step 21 — Visualisation Pipeline (`21_xgboost_curves.py`)

- Uses `xgb.train` (native API) with `early_stopping_rounds=20` to capture full training history.
- Generates publication-quality 300 DPI PNG charts.
- **Validation curves:** Log-loss convergence over boosting rounds.
- **Test curves:** ROC, Precision-Recall, Confusion Matrix, Feature Importance (Top 20).
- Saves a comprehensive `model_performance_summary.json` report.

---

## Feature Engineering

| Category | Features |
|---|---|
| **Lagged Returns** | `return_1`, `return_3`, `return_6`, `return_12`, `return_24`, `log_return_1` |
| **Simple Moving Averages** | `sma_10`, `sma_20`, `sma_50` |
| **Exponential Moving Averages** | `ema_10`, `ema_20`, `ema_50` |
| **MA Distance** | `close_vs_sma10`, `close_vs_sma20`, `close_vs_ema20` |
| **RSI** | `rsi_14`, `rsi_7` |
| **MACD** | `macd_line`, `macd_signal`, `macd_hist` |
| **Bollinger Bands** | `bb_mid`, `bb_upper`, `bb_lower`, `bb_width`, `close_vs_bb_mid` |
| **Volatility** | `volatility_12`, `volatility_24`, `volatility_48` |
| **ATR** | `atr_14`, `atr_pct_14` |
| **Price Momentum** | `momentum_3`, `momentum_6`, `momentum_12` |
| **Volume** | `volume_change_1`, `volume_sma_20`, `volume_vs_sma20` |
| **Candle Structure** | `high_low_spread`, `open_close_spread`, `candle_body`, `upper_shadow`, `lower_shadow`, `body_to_range` |

All features are computed using only **past data** at each time step, preventing look-ahead bias.

---

## Model Architecture

```
XGBoost Binary Classifier
  objective        : binary:logistic
  n_estimators     : 300
  max_depth        : 6
  learning_rate    : 0.05
  subsample        : 0.80
  colsample_bytree : 0.80
  tree_method      : hist   (fast histogram-based)
  early_stopping   : 20 rounds (Step 21 / native API)
  n_jobs           : -1
```

Sequences of length **32 candles** (= 8 hours of 15-minute bars) are flattened to a 2-D feature matrix before being passed to XGBoost. The flattened feature dimension is `32 × 48 features = 1,536`.

---

## Sequence Length vs Target Horizon

These are two **independent** parameters that control completely different things.

###  Sequence Length = *"How much past the model looks at"* (input window)

The model does not see a single candle — it receives the last **32 candles** of features as its input. This is its **memory window**.

```
Sequence Length = 32 candles = 8 hours of history

[t-31][t-30][t-29] ... [t-2][t-1][t]  ──► MODEL ──► PREDICTION
  ↑__________________________________↑
          Input window (32 candles)
     "What happened in the past 8 hours?"
```

###  Target Horizon = *"How far into the future the model predicts"* (output label)

The binary label is set **4 candles ahead** of the last input candle. This is what the model learns to predict.

```
Target Horizon = 4 candles = 1 hour ahead

                                         [t+1][t+2][t+3][t+4]
                                                            ↑
[t-31] ... [t]  ──► MODEL ──► Will close[t+4] > close[t] ?
```

### Full Picture Together

```
Timeline ──────────────────────────────────────────────────────────────────►

        Past (input window)              Now      Future (target)
        ◄──── 32 candles (8 h) ────►     │        ◄── 4 candles (1 h) ──►
                                         │
  t-31  t-30  t-29  ...  t-2  t-1  [t]  │  t+1   t+2   t+3   [t+4]
  ████  ████  ████       ████ ████ ████  │                      ████
  ↑_________________________________↑    │                        ↑
  48 features per candle                 │             Is close[t+4] > close[t]?
  fed into the model as a flat vector    │             → 1  (Up)
  of 32 × 48 = 1,536 inputs             │             → 0  (Down)
```

> **Key insight:** the sequence length and the target horizon are **completely independent**. You could look back 16 candles and predict 4 hours ahead, or look back 64 candles and predict 15 minutes ahead — they do not need to match.


## Validation Strategy

Three complementary validation approaches are used:

| Method | Description | Leakage risk |
|---|---|---|
| **Standard split (leaky)** | 70/15/15 chronological split, scaler fit on all data | ⚠ Minor |
| **No-leak split** | 70/15/15, scaler fit on train only, sequences built per-split | ✅ None |
| **Walk-forward (5-fold)** | Expanding window, each fold trained on past, tested on future | ✅ None |

The walk-forward test is the **gold standard** for time-series models and gives the most realistic estimate of live performance.

---

## Real-Time Inference Loop

Once trained, the model runs as a **lightweight real-time loop** that triggers on each new 15-minute candle close. The entire pipeline from raw data to signal takes well under 1 second — the dominant cost is the network call to Binance, not the model itself.

```
Every 15 minutes (on candle close):

  ┌─────────────────────────────────────────────────────┐
  │  1. New candle closes on Binance                    │ ~0 ms   (event)
  │  2. Fetch last 32 candles via ccxt                  │ ~200–500 ms
  │  3. Compute 48 technical features                   │ ~2 ms
  │  4. Scale with saved scaler_no_leak.pkl             │ ~0.5 ms
  │  5. Flatten → 1,536 input vector                   │ ~0.1 ms
  │  6. model.predict_proba([vector])                   │ ~5 ms
  │  7. If prob > 0.5 → signal UP, else signal DOWN     │ ~0 ms
  │  8. Send signal to execution layer                  │ ~1–10 ms
  └─────────────────────────────────────────────────────┘
  Total wall-clock time: ~300–600 ms
  Model runs: ONCE EVERY 15 MINUTES
```

> **Bottleneck:** Steps 3–8 (the actual ML work) take only **~8 ms** combined. The ~300–500 ms bottleneck is the **Binance API call** in Step 2. Switching to a persistent WebSocket feed would reduce this to < 1 ms.

| Step | Component | Latency |
|---|---|---|
| Candle close trigger | Exchange event | ~0 ms |
| Data fetch | `ccxt` REST call | ~200–500 ms |
| Feature computation | NumPy / Pandas | ~2 ms |
| Scaling | `StandardScaler.transform()` | ~0.5 ms |
| Flattening | `numpy.reshape()` | ~0.1 ms |
| **Model inference** | **`xgb.predict_proba()`** | **~5 ms** |
| Signal dispatch | REST / WebSocket | ~1–10 ms |
| **Total** | | **~300–600 ms** |

---

## Output Artifacts

```
plots/
  validation_curves/
    01_train_val_logloss.png          # Convergence history
  test_curves/
    01_roc_curve.png                  # ROC curve (AUC = 0.9063)
    02_precision_recall_curve.png     # PR tradeoff
    03_confusion_matrix.png           # Confusion matrix heatmap
    06_feature_importance.png         # Top-20 feature importance

reports/
  model_performance_summary.json      # Full metrics + model config

models/  (generated at runtime)
  xgboost_direction_model.json        # Baseline model weights
  xgboost_direction_model_no_leak.json
  xgboost_metrics.json
  xgboost_metrics_no_leak.json
  walk_forward_results.json
```

---




###  Create and activate a virtual environment

```bash
python -m venv ml_env

# Windows
ml_env\Scripts\activate

# macOS / Linux
source ml_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install ccxt numpy pandas scikit-learn xgboost matplotlib joblib
```

---

## Usage

Run the scripts **in order** from the project root directory. Each step produces the files consumed by the next.

```bash
# 1. Fetch raw OHLCV data from Binance
python 01_collect_sol_data_15m.py

# 2. Build technical indicators
python 02_build_technical_features.py

# 3. Build sequences (leaky baseline)
python 03_build_sequences.py

# 4. Split into train / val / test
python 04_split_data.py

# 5. Train baseline XGBoost
python 05_train_xgboost_baseline.py

# 7. Prepare no-leak data (recommended path)
python 07_prepare_no_leak_data.py

# 8. Train XGBoost on no-leak data
python 08_train_xgboost_no_leak.py

# 20. Run walk-forward validation
python 20_walk_forward_validation.py

# 21. Generate all curves and JSON report
python 21_xgboost_curves.py
```

> **Tip:** Step 01 can take several minutes to fetch years of 15-minute data. Subsequent steps run in seconds to a few minutes on a modern laptop.

---


### 🧠 Weather Forecaster Analogy

| Concept | Weather analogy | This project |
|---|---|---|
| **Sequence length** | You study the **last 8 hours** of weather readings | Last 32 × 15-min candles of OHLCV + indicators |
| **Target horizon** | You forecast the weather **1 hour from now** | Price direction 4 candles (1 hour) ahead |

### Why these specific values?

| Parameter | Value | Rationale |
|---|---|---|
| Sequence length | 32 candles | 8 hours of intraday memory — enough to capture momentum, consolidation, and reversal patterns without making the flattened feature matrix too large |
| Target horizon | 4 candles | 1-hour prediction — meaningful for active trading, less noisy than 1-candle (15 min), and still produces a large number of training samples |

---

## Design Decisions & Caveats

| Topic | Decision / Note |
|---|---|
| **Target horizon** | 4 candles (1 hour). Shorter horizons are noisier; longer ones reduce sample count significantly. |
| **Sequence length** | 32 candles (8 hours). Captures intraday context without ballooning the feature matrix. |
| **No shuffling** | All splits are strictly chronological to respect the time-series nature of the data. |
| **StandardScaler** | Applied after splitting (Step 07+) to prevent future statistics leaking into training. |
| **XGBoost flattening** | 3-D sequences are flattened to 2-D; XGBoost does not natively model temporal order but compensates with large feature vectors capturing lagged states. |
| **Walk-forward** | Uses expanding (not rolling) windows — each fold sees more data than the previous, mirroring real incremental learning. |
| **No live trading** | This repository is for **research and educational purposes only**. Past predictive accuracy does not guarantee future profits. Transaction costs, slippage, and regime changes are not modelled. |

