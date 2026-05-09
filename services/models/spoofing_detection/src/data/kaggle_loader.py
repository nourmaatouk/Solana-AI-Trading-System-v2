"""
kaggle_loader.py — Load Coinbase High-Frequency Crypto Data from Kaggle
=========================================================================

Loads real order book data from Kaggle and converts to SpoofingDataset format.

Features:
    - Load BTC/ETH/ADA at multiple frequencies (1sec, 1min, 5min)
    - Extract order book features (depth, imbalance, spread, etc.)
    - Generate spoofing labels using heuristics
    - Convert to SpoofingDataset for training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.data.kaggle_adapters import normalize_kaggle_crypto_frame
except ModuleNotFoundError:
    # Fallback when the file is run directly (python kaggle_loader.py)
    from kaggle_adapters import normalize_kaggle_crypto_frame

logger = logging.getLogger(__name__)


def load_kaggle_crypto_data(
    data_dir: Path = Path("data/raw/kaggle_crypto"),
    asset: str = "BTC",
    frequency: str = "1min",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Kaggle crypto limit order book data.
    
    Parameters
    ----------
    data_dir : Path to kaggle_crypto directory
    asset : "BTC", "ETH", or "ADA"
    frequency : "1sec", "1min", or "5min"
    max_rows : Max rows to load (for testing)
    
    Returns
    -------
    DataFrame with columns: midpoint, spread, bids_*, asks_*, etc.
    """
    file_path = data_dir / f"{asset}_{frequency}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    logger.info(f"Loading {asset} {frequency} data from {file_path}...")
    df = pd.read_csv(file_path, nrows=max_rows)
    df = normalize_kaggle_crypto_frame(df, n_levels=15)
    
    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns[:10])}... ({len(df.columns)} total)")
    
    return df


def extract_lob_features(
    df: pd.DataFrame,
    n_levels: int = 10,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract order book features from raw Kaggle data.
    
    Features extracted:
        - Midpoint price
        - Spread (ask - bid)
        - Order imbalance (bid_vol / ask_vol)
        - Depth levels (1-10)
        - Volume profiles
        - Volatility indicators
    
    Parameters
    ----------
    df : Raw Kaggle DataFrame
    n_levels : Number of LOB depth levels (max 15 in Kaggle data)
    window_length : Sequence length (100 events)
    
    Returns
    -------
    X : (n_sequences, window_length, n_features) feature array
    df_features : DataFrame with per-tick features
    """
    logger.info(f"Extracting {n_levels} LOB levels features...")

    features_list = []

    # Extract features from each row
    for _, row in df.iterrows():
        tick_features = []
        
        # 1. Midpoint and spread
        midpoint = row.get('midpoint', row.get('mid_price', 0))
        spread = row.get('spread', 0)
        
        tick_features.append(midpoint)
        tick_features.append(spread)
        
        # 2. Bid/Ask volumes at each level
        for level in range(1, n_levels + 1):
            bid_col = f'bids_limit_notional_{level}'
            ask_col = f'asks_limit_notional_{level}'
            
            if bid_col in row and ask_col in row:
                bid_vol = float(row[bid_col]) if pd.notna(row[bid_col]) else 0
                ask_vol = float(row[ask_col]) if pd.notna(row[ask_col]) else 0
                
                tick_features.append(bid_vol)
                tick_features.append(ask_vol)
        
        # 3. Order imbalance (normalized)
        total_bid = sum(float(row.get(f'bids_limit_notional_{i}', 0)) or 0 for i in range(1, n_levels + 1))
        total_ask = sum(float(row.get(f'asks_limit_notional_{i}', 0)) or 0 for i in range(1, n_levels + 1))
        
        if total_bid + total_ask > 0:
            imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        else:
            imbalance = 0
        
        tick_features.append(imbalance)
        
        # Pad to ensure consistent feature count
        n_features_expected = 2 + (n_levels * 2) + 1  # midpoint, spread, bid/ask per level, imbalance
        while len(tick_features) < n_features_expected:
            tick_features.append(0)
        
        features_list.append(tick_features[:n_features_expected])
    
    X_features = np.array(features_list, dtype=np.float32)

    # Normalize: replace NaN/Inf from edge rows, then StandardScaler
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features).astype(np.float32)

    logger.info(f"Extracted and normalized shape: {X_features.shape}")

    return X_features, df


def label_real_spoofing(
    df: pd.DataFrame,
    features: np.ndarray,
    window_length: int = 100,
    window_stride: int = 50,
    n_levels: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spoofing labels using adaptive heuristics for real data.
    
    Spoofing indicators (based on market microstructure):
        1. High cancel-to-total ratio (>65th percentile)
        2. Extreme order imbalance (>70th percentile)
        3. Tight spreads (potential layering)
    
    Parameters
    ----------
    df : DataFrame with raw data
    features : Extracted features array
    window_length : Sequence window size
    window_stride : Stride for sliding window
    
    Returns
    -------
    X : (n_windows, window_length, n_features)
    y : (n_windows,) binary labels (target ~30% spoofing)
    """
    logger.info("Labeling spoofing patterns (adaptive thresholds)...")
    
    spoofing_scores = []
    
    for _, row in df.iterrows():
        score = 0.0

        # Feature 1: Cancel-to-total ratio
        cancel_vol = sum(float(row.get(f'bids_cancel_notional_{level}', 0) or 0) for level in range(1, n_levels + 1))
        cancel_vol += sum(float(row.get(f'asks_cancel_notional_{level}', 0) or 0) for level in range(1, n_levels + 1))
        limit_vol = sum(float(row.get(f'bids_limit_notional_{level}', 0) or 0) for level in range(1, n_levels + 1))
        limit_vol += sum(float(row.get(f'asks_limit_notional_{level}', 0) or 0) for level in range(1, n_levels + 1))
        
        if limit_vol > 0:
            cancel_ratio = cancel_vol / (limit_vol + cancel_vol)
            score += cancel_ratio  # Directly use ratio as score
        
        # Feature 2: Order imbalance
        buys = float(row.get('buys', 0) or 0)
        sells = float(row.get('sells', 0) or 0)
        
        if buys + sells > 0:
            imbalance = abs(buys - sells) / (buys + sells)
            score += imbalance * 0.5  # Weight it less than cancel ratio
        
        spoofing_scores.append(score)
    
    spoofing_scores = np.array(spoofing_scores)

    # Rolling 70th-percentile threshold (window=500 rows, min 50 to warm up).
    # A global threshold concentrates positives in high-volatility periods and
    # leaves the chronological test split nearly empty — train ≈47% vs test ≈2%.
    # The rolling threshold keeps the positive rate stable (~30%) across time so
    # all three splits have a comparable density of labelled spoofing events.
    _ROLL = 500
    scores_s = pd.Series(spoofing_scores)
    roll_thresh = (
        scores_s
        .rolling(_ROLL, min_periods=50)
        .quantile(0.70)
    )
    # Back-fill the first <50 rows with the expanding quantile
    expanding_thresh = scores_s.expanding(min_periods=1).quantile(0.70)
    roll_thresh = roll_thresh.fillna(expanding_thresh)
    is_spoof = spoofing_scores > roll_thresh.values

    logger.info(f"Rolling threshold range: {roll_thresh.min():.4f} – {roll_thresh.max():.4f}")
    logger.info(f"Spoofing rate before windowing: {is_spoof.mean():.1%}")
    
    # Create sliding windows
    X_windows = []
    y_windows = []
    
    n_windows = (len(features) - window_length) // window_stride + 1
    
    for i in range(n_windows):
        start_idx = i * window_stride
        end_idx = start_idx + window_length
        
        if end_idx <= len(features):
            X_windows.append(features[start_idx:end_idx])
            
            # Label: 1 if >40% of events in window are spoofing
            window_spoof_count = np.sum(is_spoof[start_idx:end_idx])
            window_label = int(window_spoof_count >= window_length * 0.4)
            y_windows.append(window_label)
    
    X_windowed = np.array(X_windows, dtype=np.float32)
    y_windowed = np.array(y_windows, dtype=np.int64)
    
    logger.info(f"Created {len(X_windows)} windows")
    logger.info(f"Spoofing labels: {y_windowed.sum()} positive ({100*y_windowed.mean():.1f}%)")
    
    return X_windowed, y_windowed


def prepare_kaggle_data(
    asset: str = "BTC",
    frequency: str = "1min",
    max_rows: Optional[int] = None,
    data_dir: Path = Path("data/raw/kaggle_crypto"),
    n_levels: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: Load → Extract → Label Kaggle data.
    
    Parameters
    ----------
    asset : "BTC", "ETH", "ADA"
    frequency : "1sec", "1min", "5min"
    max_rows : Limit rows for testing
    data_dir : Path to kaggle data
    
    Returns
    -------
    X : (n_sequences, 100, n_features)
    y : (n_sequences,)
    """
    # Load
    df = load_kaggle_crypto_data(data_dir, asset, frequency, max_rows)
    
    # Extract features
    features, _ = extract_lob_features(df, n_levels=n_levels)
    
    # Label and window
    X, y = label_real_spoofing(df, features, window_length=100, window_stride=50, n_levels=n_levels)
    
    logger.info(f"\n✓ Prepared {asset} {frequency}")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Spoofing: {y.sum()} ({100*y.mean():.1f}%)")
    
    return X, y


if __name__ == "__main__":
    import logging as log_config
    log_config.basicConfig(level=log_config.INFO)
    
    # Example: Load BTC 1-minute data
    X, y = prepare_kaggle_data(
        asset="BTC",
        frequency="1min",
        max_rows=5000,  # Test with first 5000 rows
    )
    
    print(f"\n✓ Successfully loaded Kaggle data!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive rate: {y.mean():.1%}")
