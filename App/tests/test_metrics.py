import os
import pytest
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_engine")))
from metrics import compute_point_adjusted_metrics

def test_point_adjusted_metrics():
    N = 100
    y_true = np.zeros(N, dtype=bool)
    y_pred = np.zeros(N, dtype=bool)

    # Ground truth event range [20, 30]
    y_true[20:31] = True
    # Single prediction hit at index 25
    y_pred[25] = True

    metrics = compute_point_adjusted_metrics(y_true, y_pred, ground_truth_seqs=[[20, 30]])

    # Because index 25 hit, the point-adjusted recall should be 1.0
    assert metrics["event_recall"] == 1.0
    assert metrics["point_adjusted_f1"] > 0.9
    assert metrics["false_alarm_rate"] == 0.0
