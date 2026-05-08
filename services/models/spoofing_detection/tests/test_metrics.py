import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_point_metrics


def test_compute_point_metrics_accepts_python_lists():
    metrics = compute_point_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 4) == 0.6667


def test_compute_point_metrics_accepts_class_probability_matrix():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.7, 0.3],
        ]
    )

    metrics = compute_point_metrics(y_true, y_prob)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_point_metrics_thresholds_spoofing_probabilities():
    metrics = compute_point_metrics([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.3])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
