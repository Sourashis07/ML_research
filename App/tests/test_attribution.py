import os
import pytest
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_engine")))
from attribution import RootCauseAttributor

def test_channel_attribution_ranking():
    attributor = RootCauseAttributor(channel_names=["V_bus", "I_sa", "T_batt", "T_bay"])
    
    # Injected massive error in channel 2 (T_batt)
    x_actual = np.array([0.5, 0.5, 2.5, 0.5])
    x_recon  = np.array([0.5, 0.5, 0.5, 0.5])

    top_causes = attributor.get_top_k_root_causes(x_actual, x_recon, top_k=2)

    assert len(top_causes) == 2
    assert top_causes[0]["channel_index"] == 2
    assert top_causes[0]["channel_name"] == "T_batt"
    assert top_causes[0]["contribution_score"] > 0.8
