"""
Shared ML training utilities.

Centralises calibrate_threshold which was previously duplicated in both
08_train_xgboost_no_leak.py and 20_walk_forward_validation.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def calibrate_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Sweep thresholds on the validation set and return the one that maximises F1.

    Searches the range [0.05, 0.95] at 0.005 increments (181 candidates).
    Prints the best threshold and its F1 score.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = np.array([
        f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ])
    best_thresh = float(thresholds[f1s.argmax()])
    print(f"  Val-F1 calibration: best threshold = {best_thresh:.3f} "
          f"(F1 = {f1s.max():.4f})")
    return best_thresh
