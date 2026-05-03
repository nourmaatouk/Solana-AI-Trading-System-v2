import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample
from typing import Dict, Tuple

def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute base classification metrics."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0))
    }

def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 1000) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% Confidence Intervals for metrics."""
    stats = {"precision": [], "recall": [], "f1": []}
    
    n = len(y_true)
    if n == 0:
        return {k: (0.0, 0.0) for k in stats}
        
    for _ in range(n_resamples):
        # We must stratify if possible
        if len(np.unique(y_true)) < 2:
            indices = np.random.randint(0, n, n)
        else:
            indices = resample(np.arange(n), stratify=y_true)
            
        y_t_b = y_true[indices]
        y_p_b = y_pred[indices]
        m = compute_point_metrics(y_t_b, y_p_b)
        
        for k in stats:
            stats[k].append(m[k])
            
    ci = {}
    for k, v in stats.items():
        v_arr = np.array(v)
        ci[k] = (float(np.percentile(v_arr, 2.5)), float(np.percentile(v_arr, 97.5)))
        
    return ci

def compute_detection_delay_ms(event_times_ms: np.ndarray, predictions: np.ndarray, spoofing_start_ms: float) -> float:
    """
    Computes time from true spoofing sequence start to first confident prediction > 0.5.
    
    Parameters
    ----------
    event_times_ms : np.ndarray
        Timestamps of each sliding window evaluation step.
    predictions : np.ndarray
        Spoofing probability predictions (0 to 1).
    spoofing_start_ms : float
        Actual timestamp when spoofing injection started.
        
    Returns
    -------
    delay_ms : float
        Delay in milliseconds. Returns np.nan if never confidently detected after start.
    """
    valid_mask = event_times_ms >= spoofing_start_ms
    valid_events = event_times_ms[valid_mask]
    valid_preds = predictions[valid_mask]
    
    confident_idx = np.where(valid_preds > 0.5)[0]
    
    if len(confident_idx) == 0:
        return float('nan')
        
    first_detect_time = valid_events[confident_idx[0]]
    delay = first_detect_time - spoofing_start_ms
    
    return float(delay)
