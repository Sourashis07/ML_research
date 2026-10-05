import os
import pytest
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_engine")))
from thresholding import DynamicThresholdEngine

def test_dynamic_threshold_calculation():
    engine = DynamicThresholdEngine(n_sigma=3.0)
    
    # Nominal errors around 0.1, anomaly spike at indices 200..250 around 1.5
    N = 500
    x_actual = np.random.normal(0.5, 0.05, size=(N, 5))
    x_recon = x_actual.copy()
    x_actual[200:250, 0] += 1.5  # Inject anomaly spike into channel 0

    results = engine.detect_anomalies(x_actual, x_recon)

    assert "threshold" in results
    assert "anomaly_mask" in results
    assert results["threshold"] > 0.1
    # Check that injected anomaly range [200, 250] is detected
    anom_detected = np.where(results["anomaly_mask"])[0]
    assert len(anom_detected) > 0
    assert np.any((anom_detected >= 200) & (anom_detected <= 250))
