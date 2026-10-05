import numpy as np
import pandas as pd

class DynamicThresholdEngine:
    """
    Non-Parametric Dynamic Thresholding (NDT) Engine for NASA Telemanom.
    Computes smoothed residual errors e_t = |x_t - x_hat_t|, calculates EWMA,
    and determines dynamic anomaly threshold bounds based on mu + 3*sigma or NDT ratio optimization.
    """
    def __init__(self, ewma_alpha: float = 0.1, n_sigma: float = 3.0):
        self.ewma_alpha = ewma_alpha
        self.n_sigma = n_sigma

    def compute_residuals(self, x_actual: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
        """
        Computes pointwise reconstruction residual absolute errors.
        x_actual, x_recon: [N, C] or [N]
        """
        residuals = np.abs(x_actual - x_recon)
        return residuals

    def ewma_smooth(self, residuals: np.ndarray) -> np.ndarray:
        """
        Applies Exponentially Weighted Moving Average (EWMA) smoothing over residual sequence.
        """
        if residuals.ndim == 1:
            series = pd.Series(residuals)
            return series.ewm(alpha=self.ewma_alpha).mean().values
        else:
            smoothed = np.zeros_like(residuals)
            for c in range(residuals.shape[1]):
                series = pd.Series(residuals[:, c])
                smoothed[:, c] = series.ewm(alpha=self.ewma_alpha).mean().values
            return smoothed

    def find_dynamic_threshold(self, smoothed_residuals: np.ndarray) -> float:
        """
        Computes non-parametric dynamic threshold epsilon for a single channel sequence.
        Formula: epsilon = mu(e_t) + n_sigma * sigma(e_t)
        Can also perform adaptive pruning search if error variance is high.
        """
        if smoothed_residuals.ndim > 1:
            # Focus on primary target channel 0
            err_seq = smoothed_residuals[:, 0]
        else:
            err_seq = smoothed_residuals

        mu = np.mean(err_seq)
        sigma = np.std(err_seq)
        threshold = mu + self.n_sigma * sigma

        # Search for threshold that maximizes standard deviation drop (NDT approach)
        best_threshold = threshold
        max_score = 0.0

        for s in np.linspace(2.5, 4.5, 20):
            cand_thresh = mu + s * sigma
            anom_indices = np.where(err_seq > cand_thresh)[0]
            if len(anom_indices) == 0:
                continue

            # Calculate reduction ratio
            clean_seq = np.delete(err_seq, anom_indices)
            if len(clean_seq) == 0:
                continue
            delta_mu = (mu - np.mean(clean_seq)) / mu if mu > 0 else 0
            delta_sigma = (sigma - np.std(clean_seq)) / sigma if sigma > 0 else 0
            score = (delta_mu + delta_sigma) / (len(anom_indices) + 1e-5)

            if score > max_score:
                max_score = score
                best_threshold = cand_thresh

        return float(best_threshold)

    def detect_anomalies(self, x_actual: np.ndarray, x_recon: np.ndarray) -> dict:
        """
        Full anomaly detection execution.
        Returns dictionary containing:
        - raw_residuals
        - smoothed_residuals
        - threshold
        - anomaly_mask (boolean array)
        - anomaly_sequences (list of [start, end] pairs)
        """
        raw_res = self.compute_residuals(x_actual, x_recon)
        smoothed_res = self.ewma_smooth(raw_res)

        # Primary target channel residual (col 0)
        target_res = smoothed_res[:, 0] if smoothed_res.ndim > 1 else smoothed_res
        threshold = self.find_dynamic_threshold(smoothed_res)

        anomaly_mask = target_res > threshold

        # Group consecutive True flags into sequence interval ranges [start, end]
        anomaly_seqs = []
        in_anom = False
        start = 0

        for i, val in enumerate(anomaly_mask):
            if val and not in_anom:
                in_anom = True
                start = i
            elif not val and in_anom:
                in_anom = False
                anomaly_seqs.append([start, i - 1])

        if in_anom:
            anomaly_seqs.append([start, len(anomaly_mask) - 1])

        return {
            "raw_residuals": raw_res,
            "smoothed_residuals": smoothed_res,
            "threshold": threshold,
            "anomaly_mask": anomaly_mask,
            "anomaly_sequences": anomaly_seqs
        }
