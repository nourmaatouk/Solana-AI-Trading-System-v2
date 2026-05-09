import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from data.kaggle_adapters import normalize_kaggle_crypto_frame


def test_normalize_kaggle_crypto_frame_maps_aliases():
    df = pd.DataFrame([
        {
            "timestamp": "2026-04-30T10:00:00Z",
            "mid_price": 101.5,
            "bids_notional_0": 10,
            "asks_notional_0": 12,
            "bids_cancel_notional_0": 1,
            "asks_cancel_notional_0": 2,
            "buys": 7,
            "sells": 5,
        }
    ])

    normalized = normalize_kaggle_crypto_frame(df, n_levels=1)

    assert not normalized.empty
    assert pd.api.types.is_datetime64_any_dtype(normalized["timestamp"])
    assert normalized.iloc[0]["midpoint"] == 101.5
    assert normalized.iloc[0]["bids_limit_notional_1"] == 10
    assert normalized.iloc[0]["asks_limit_notional_1"] == 12
    assert normalized.iloc[0]["bids_cancel_notional_1"] == 1
    assert normalized.iloc[0]["asks_cancel_notional_1"] == 2
    assert normalized.iloc[0]["buys"] == 7
    assert normalized.iloc[0]["sells"] == 5
