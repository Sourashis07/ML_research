import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TelemetryScaler:
    """
    Min-Max Scaler fitted strictly on nominal training telemetry [N, C].
    Ensures zero data leakage from test sequences.
    """
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        """Fit scaler on nominal telemetry array [N, C]."""
        self.scaler.fit(data)
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform array using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("TelemetryScaler must be fitted on training data first!")
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform_col0(self, scaled_col0: np.ndarray) -> np.ndarray:
        """Inverse transform for target continuous telemetry channel (Col 0)."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted.")
        min_val = self.scaler.data_min_[0]
        max_val = self.scaler.data_max_[0]
        return scaled_col0 * (max_val - min_val) + min_val


class TelemetryWindowDataset(Dataset):
    """
    Sliding Window PyTorch Dataset for Spacecraft Telemetry.
    Inputs:
        data: Array of shape [N, C] where Col 0 = Continuous Target, Cols 1+ = Command vectors.
        window_size: Sequence window size W (default 100).
        stride: Step stride s (default 1).
    Returns:
        x_window: Tensor of shape [W, C] representing input sequence window.
        y_target: Tensor of shape [C] representing next timestep values or full window reconstruction target.
    """
    def __init__(self, data: np.ndarray, window_size: int = 100, stride: int = 1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.stride = stride

        num_steps = self.data.shape[0]
        if num_steps < window_size:
            # Pad if shorter than window size
            padding = torch.zeros((window_size - num_steps, self.data.shape[1]), dtype=torch.float32)
            self.data = torch.cat([padding, self.data], dim=0)
            num_steps = window_size

        self.num_windows = (num_steps - window_size) // stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        window = self.data[start_idx:end_idx] # [W, C]
        return window, window  # Autoencoder target is reconstruction of input window


def create_telemetry_dataloaders(train_data: np.ndarray, test_data: np.ndarray, 
                                 window_size: int = 100, batch_size: int = 64, stride: int = 1):
    """
    Helper function to scale data strictly on train_data, generate sliding windows,
    and return PyTorch DataLoaders + Scaler instance.
    """
    scaler = TelemetryScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    train_dataset = TelemetryWindowDataset(scaled_train, window_size=window_size, stride=stride)
    test_dataset = TelemetryWindowDataset(scaled_test, window_size=window_size, stride=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, scaler, scaled_train, scaled_test
