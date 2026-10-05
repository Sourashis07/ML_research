import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TelemetryScaler, TelemetryWindowDataset, create_telemetry_dataloaders
from model import SpatialTemporalAutoencoder
from thresholding import DynamicThresholdEngine
from attribution import RootCauseAttributor
from metrics import compute_point_adjusted_metrics

class TelemetryAIPipeline:
    """
    End-to-end Machine Learning Pipeline for Spacecraft Telemetry Anomaly Detection.
    """
    def __init__(self, num_channels: int, window_size: int = 100, hidden_dim: int = 64, lr: float = 1e-3):
        self.num_channels = num_channels
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SpatialTemporalAutoencoder(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            num_heads=4
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.threshold_engine = DynamicThresholdEngine()
        self.attributor = RootCauseAttributor()
        self.scaler = None

    def train_model(self, train_data: np.ndarray, epochs: int = 10, batch_size: int = 64):
        """Trains SpatialTemporalAutoencoder on nominal training sequence."""
        self.scaler = TelemetryScaler()
        scaled_train = self.scaler.fit_transform(train_data)
        dataset = TelemetryWindowDataset(scaled_train, window_size=self.window_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = self.criterion(recon, batch_x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch_x)

            avg_loss = total_loss / len(dataset)
            if epoch % 5 == 0 or epoch == epochs:
                print(f"[Train] Epoch {epoch:02d}/{epochs:02d} | Reconstruction MSE Loss: {avg_loss:.6f}")

        return self

    def predict_sequence(self, test_data: np.ndarray) -> dict:
        """
        Runs model reconstruction on test telemetry and returns actual vs reconstructed sequences,
        threshold bounds, anomaly masks, and channel root-cause attributions.
        """
        if self.scaler is None:
            self.scaler = TelemetryScaler().fit(test_data)

        scaled_test = self.scaler.transform(test_data)
        dataset = TelemetryWindowDataset(scaled_test, window_size=self.window_size, stride=1)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        self.model.eval()
        recon_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                recon, _ = self.model(batch_x)
                # Extract reconstructed last step of each window to map 1:1 to timesteps
                recon_list.append(recon[:, -1, :].cpu().numpy())

        recon_scaled = np.vstack(recon_list)

        # Truncate scaled test to match reconstruction length
        scaled_test_matched = scaled_test[self.window_size - 1:]
        test_data_matched = test_data[self.window_size - 1:]

        # Inverse transform Col 0 reconstruction to original scale for UI plot
        recon_col0_orig = self.scaler.inverse_transform_col0(recon_scaled[:, 0])

        # Dynamic Thresholding on scaled residuals
        anom_results = self.threshold_engine.detect_anomalies(scaled_test_matched, recon_scaled)

        # Top root-cause channel attributions
        df_attribution = self.attributor.compute_attribution(scaled_test_matched, recon_scaled)

        return {
            "x_actual_orig": test_data_matched[:, 0],
            "x_recon_orig": recon_col0_orig,
            "scaled_actual": scaled_test_matched,
            "scaled_recon": recon_scaled,
            "raw_residuals": anom_results["raw_residuals"],
            "smoothed_residuals": anom_results["smoothed_residuals"],
            "threshold": anom_results["threshold"],
            "anomaly_mask": anom_results["anomaly_mask"],
            "anomaly_sequences": anom_results["anomaly_sequences"],
            "df_attribution": df_attribution
        }

def run_pipeline_demo(data_dir="App/data", chan_id="P-1", epochs=5):
    """Execution wrapper for pipeline training and evaluation on a target channel."""
    train_path = os.path.join(data_dir, "train", f"{chan_id}.npy")
    test_path = os.path.join(data_dir, "test", f"{chan_id}.npy")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[!] Target channel file '{chan_id}.npy' not found in '{data_dir}'.")
        return None

    train_arr = np.load(train_path)
    test_arr = np.load(test_path)
    num_channels = train_arr.shape[1]

    print(f"[+] Initializing ML Engine for Channel '{chan_id}' ({num_channels} telemetry channels)...")
    pipeline = TelemetryAIPipeline(num_channels=num_channels, hidden_dim=32)
    pipeline.train_model(train_arr, epochs=epochs)

    results = pipeline.predict_sequence(test_arr)

    # Evaluate against labeled anomalies if CSV available
    csv_path = os.path.join(data_dir, "labeled_anomalies.csv")
    if os.path.exists(csv_path):
        df_lbl = pd.read_csv(csv_path)
        chan_meta = df_lbl[df_lbl["chan_id"] == chan_id]
        if not chan_meta.empty:
            gt_seqs = json.loads(chan_meta.iloc[0]["anomaly_sequences"])
            N = len(results["anomaly_mask"])
            gt_mask = np.zeros(N, dtype=bool)
            for s, e in gt_seqs:
                # Adjust for window offset
                adj_s = max(0, s - 99)
                adj_e = max(0, e - 99)
                gt_mask[adj_s:adj_e+1] = True

            eval_metrics = compute_point_adjusted_metrics(gt_mask, results["anomaly_mask"], ground_truth_seqs=gt_seqs)
            results["metrics"] = eval_metrics
            print(f"[+] Benchmark Results for {chan_id}: Point-Adjusted F1={eval_metrics['f1']:.4f}, Recall={eval_metrics['recall']:.4f}, FAR={eval_metrics['false_alarm_rate']:.4f}")

    return results

if __name__ == "__main__":
    app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(app_root, "data")
    run_pipeline_demo(data_dir=data_folder, chan_id="P-1", epochs=5)
