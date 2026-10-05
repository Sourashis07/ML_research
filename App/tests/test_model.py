import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import numpy as np
import sys

# Add ml_engine to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_engine")))
from model import SpatialTemporalAutoencoder

def test_model_forward_pass():
    batch_size = 8
    window_size = 100
    num_channels = 25

    model = SpatialTemporalAutoencoder(num_channels=num_channels, hidden_dim=32, num_heads=4)
    dummy_input = torch.randn(batch_size, window_size, num_channels)

    recon, attn_weights = model(dummy_input)

    assert recon.shape == (batch_size, window_size, num_channels), f"Expected shape {(batch_size, window_size, num_channels)}, got {recon.shape}"
    assert not torch.isnan(recon).any(), "Model reconstruction contains NaNs"

def test_model_gradient_flow():
    model = SpatialTemporalAutoencoder(num_channels=10, hidden_dim=16, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dummy_input = torch.randn(4, 50, 10)

    recon, _ = model(dummy_input)
    loss = torch.mean((recon - dummy_input) ** 2)
    loss.backward()

    optimizer.step()
    assert loss.item() >= 0.0
