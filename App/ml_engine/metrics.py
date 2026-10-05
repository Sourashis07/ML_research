import numpy as np

def compute_point_adjusted_metrics(y_true_mask: np.ndarray, y_pred_mask: np.ndarray, 
                                   ground_truth_seqs: list = None) -> dict:
    """
    Computes Point-Adjusted F1, Event-based Precision, Recall, and False Alarm Rate (FAR).
    - y_true_mask: boolean array of ground truth anomalies [N]
    - y_pred_mask: boolean array of predicted anomalies [N]
    - ground_truth_seqs: list of ground truth anomaly intervals [[start, end], ...]
    """
    N = len(y_true_mask)
    if N == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "far": 0.0, "point_adjusted_f1": 0.0}

    # Extract ground truth sequence intervals if not directly provided
    if ground_truth_seqs is None:
        ground_truth_seqs = []
        in_seq = False
        s = 0
        for i, val in enumerate(y_true_mask):
            if val and not in_seq:
                in_seq = True
                s = i
            elif not val and in_seq:
                in_seq = False
                ground_truth_seqs.append([s, i - 1])
        if in_seq:
            ground_truth_seqs.append([s, N - 1])

    # Point-adjusted prediction mask
    y_pred_adj = np.array(y_pred_mask, dtype=bool).copy()

    # Event-based statistics
    tp_events = 0
    fn_events = 0
    total_gt_events = len(ground_truth_seqs)

    for start, end in ground_truth_seqs:
        # Check if any detection falls within [start, end]
        if np.any(y_pred_mask[start:end + 1]):
            tp_events += 1
            y_pred_adj[start:end + 1] = True # Propagate detection across whole interval
        else:
            fn_events += 1

    # Point-level statistics after adjustment
    tp = np.sum(y_pred_adj & y_true_mask)
    fp = np.sum(y_pred_adj & (~y_true_mask))
    fn = np.sum((~y_pred_adj) & y_true_mask)
    tn = np.sum((~y_pred_adj) & (~y_true_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    event_recall = tp_events / total_gt_events if total_gt_events > 0 else 1.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "point_adjusted_f1": float(f1),
        "event_recall": float(event_recall),
        "false_alarm_rate": float(far),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
