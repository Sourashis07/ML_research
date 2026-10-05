import numpy as np
import pandas as pd

class RootCauseAttributor:
    """
    Channel-Level Explainability & Root-Cause Attribution Engine.
    Computes normalized contribution scores for each subsystem channel c at timestep t:
    Contribution(c) = |x_{t, c} - x_hat_{t, c}| / sum_j |x_{t, j} - x_hat_{t, j}|
    """
    def __init__(self, channel_names=None):
        self.channel_names = channel_names

    def _get_channel_label(self, idx: int) -> str:
        if self.channel_names and idx < len(self.channel_names):
            return self.channel_names[idx]
        if idx == 0:
            return "CH_0 (Telemetry Target)"
        return f"CH_{idx} (Cmd / Sensor {idx})"

    def compute_attribution(self, x_actual: np.ndarray, x_recon: np.ndarray, timestep: int = -1) -> pd.DataFrame:
        """
        Computes channel contribution breakdown at a specific timestep or averaged over an anomaly window.
        x_actual, x_recon: arrays of shape [N, C] or [C]
        """
        if x_actual.ndim == 1:
            abs_err = np.abs(x_actual - x_recon)
        elif timestep >= 0:
            abs_err = np.abs(x_actual[timestep] - x_recon[timestep])
        else:
            # Average absolute error across all timesteps
            abs_err = np.mean(np.abs(x_actual - x_recon), axis=0)

        total_err = np.sum(abs_err) + 1e-8
        contributions = abs_err / total_err

        records = []
        for c in range(len(contributions)):
            records.append({
                "channel_index": c,
                "channel_name": self._get_channel_label(c),
                "absolute_error": float(abs_err[c]),
                "contribution_score": float(contributions[c]),
                "contribution_pct": float(contributions[c] * 100.0)
            })

        df = pd.DataFrame(records)
        df = df.sort_values(by="contribution_score", ascending=False).reset_index(drop=True)
        return df

    def get_top_k_root_causes(self, x_actual: np.ndarray, x_recon: np.ndarray, 
                              timestep: int = -1, top_k: int = 5) -> list:
        """
        Returns top-k failing subsystem channels with root-cause attribution scores.
        """
        df_attr = self.compute_attribution(x_actual, x_recon, timestep=timestep)
        return df_attr.head(top_k).to_dict(orient="records")
