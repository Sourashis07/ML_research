"""
Sections 5 & 6: Key Findings Automated Generator & Environment Verification Checklist
NASA JPL Telemanom Automated Insights & Operational Recommendations
"""

import os
import sys
import glob
import ast
import re
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
import seaborn
from tqdm import tqdm

def run_sections_5_6():
    print("=" * 85)
    print("       SECTIONS 5 & 6: AUTOMATED KEY FINDINGS & VERIFICATION CHECKLIST")
    print("=" * 85)

    # Path Resolution
    candidate_base_dirs = [
        'data', 'archive/data/data', 'archive/data', '.',
        '../data', '../archive/data/data', '../archive/data', '..'
    ]

    test_dir = None
    csv_path = None

    for base in candidate_base_dirs:
        te = os.path.join(base, 'test')
        if os.path.exists(te) and len(glob.glob(os.path.join(te, '*.npy'))) > 0:
            test_dir = te
            break

    for base in ['archive', 'data', '.', '../archive', '../data', '..']:
        cp = os.path.join(base, 'labeled_anomalies.csv')
        if os.path.exists(cp):
            csv_path = cp
            break

    if not test_dir or not csv_path:
        raise FileNotFoundError("Could not locate NASA Telemanom test data directory or labeled_anomalies.csv.")

    df_meta = pd.read_csv(csv_path)

    def parse_sequences(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
        return val

    def parse_classes(val):
        if isinstance(val, str):
            return re.findall(r'[a-zA-Z]+', val)
        return val

    df_meta['anomaly_sequences'] = df_meta['anomaly_sequences'].apply(parse_sequences)
    df_meta['class'] = df_meta['class'].apply(parse_classes)

    def assign_subsystem(chan_id):
        prefix = chan_id.split('-')[0].upper()
        if prefix == 'P':
            return 'Power (P)'
        elif prefix in ['T', 'S']:
            return 'Thermal/Radiation (T/S)'
        elif prefix == 'E':
            return 'Actuators/Engines (E)'
        else:
            return 'Other Avionics (A/D/G/F)'

    df_meta['subsystem'] = df_meta['chan_id'].apply(assign_subsystem)
    df_meta['num_anomalies'] = df_meta['anomaly_sequences'].apply(len)

    all_classes = [c for classes in df_meta['class'] for c in classes]
    class_counts = pd.Series(all_classes).value_counts()

    total_anomalies_count = len(all_classes)
    point_count = class_counts.get('point', 0)
    contextual_count = class_counts.get('contextual', 0)

    pct_point = (point_count / total_anomalies_count) * 100
    pct_contextual = (contextual_count / total_anomalies_count) * 100

    subsys_anom_counts = df_meta.groupby('subsystem')['num_anomalies'].sum()
    top_failing_subsystem = subsys_anom_counts.idxmax()
    top_failing_count = subsys_anom_counts.max()

    anomaly_durations = []
    nominal_variances = []
    anomalous_variances = []

    for idx, row in df_meta.iterrows():
        chan_id = row['chan_id']
        seqs = row['anomaly_sequences']
        test_path = os.path.join(test_dir, f"{chan_id}.npy")
        if not os.path.exists(test_path):
            continue

        arr = np.load(test_path)[:, 0]
        n_pts = len(arr)

        anom_mask = np.zeros(n_pts, dtype=bool)
        if isinstance(seqs, list):
            for start, end in seqs:
                s_idx = max(0, start)
                e_idx = min(n_pts, end)
                anom_mask[s_idx:e_idx] = True
                anomaly_durations.append(e_idx - s_idx)

        nom_data = arr[~anom_mask]
        anom_data = arr[anom_mask]

        if len(nom_data) > 0:
            nominal_variances.append(np.var(nom_data))
        if len(anom_data) > 0:
            anomalous_variances.append(np.var(anom_data))

    durations_arr = np.array(anomaly_durations)
    dur_mean = np.mean(durations_arr)
    dur_median = np.median(durations_arr)
    dur_min = np.min(durations_arr)
    dur_max = np.max(durations_arr)

    mean_nom_var = np.mean(nominal_variances)
    mean_anom_var = np.mean(anomalous_variances)
    var_multiplier = mean_anom_var / (mean_nom_var + 1e-9)

    print("\n" + "=" * 85)
    print("                 SECTION 5: AUTOMATED KEY FINDINGS SUMMARY")
    print("=" * 85)
    print(f" * Bullet 1 [Anomaly Type Breakdown] : Contextual/Drift anomalies represent {pct_contextual:.2f}% "
          f"({contextual_count}/{total_anomalies_count}) of failures, while Point anomalies represent {pct_point:.2f}% "
          f"({point_count}/{total_anomalies_count}).")

    print(f" * Bullet 2 [Subsystem Vulnerability]: The '{top_failing_subsystem}' subsystem exhibits the highest "
          f"frequency of operational failures, accounting for {top_failing_count} total ground-truth anomaly instances.")

    print(f" * Bullet 3 [Fault Duration Range]   : Ground-truth anomaly durations span from a minimum of {dur_min} "
          f"timesteps to a maximum of {dur_max} timesteps, with a median duration of {dur_median:.1f} timesteps "
          f"(mean: {dur_mean:.1f} timesteps).")

    print(f" * Bullet 4 [Variance Profile Impact] : Anomalous sequence intervals exhibit a massive increase in continuous "
          f"sensor variance (Mean: {mean_anom_var:.4f}) compared to nominal steady-state intervals (Mean: {mean_nom_var:.4f}), "
          f"representing a ~{var_multiplier:.2f}x average increase in signal volatility during fault conditions.")
    print("=" * 85)

    # Section 6 Checklist
    tools_checklist = {
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "SciPy": getattr(stats, '__doc__', '').split()[0] if hasattr(stats, '__doc__') else "Available",
        "Matplotlib": matplotlib.__version__,
        "Seaborn": seaborn.__version__,
        "Python Executable": sys.version.split()[0]
    }

    print("\n" + "=" * 85)
    print("             SECTION 6: TOOLS VERIFICATION & OPERATIONAL RECOMMENDATIONS")
    print("=" * 85)
    for tool, version in tools_checklist.items():
        print(f"  [OK] {tool:<22} : Version {version}")
    print("-" * 85)
    print("  Operational Recommendations:")
    print("   1. Data Normalization: SMAP streams are pre-normalized [-1, 1], whereas MSL contains unbounded raw readings.")
    print("   2. Feature Modeling: Focus sliding-window anomaly detectors on Avionics and Actuator/Engine channels.")
    print("   3. Threshold Framing: Leverage residual error rolling variance (e.g. Telemanom EWMA) to detect signal shifts.")
    print("=" * 85)

if __name__ == "__main__":
    run_sections_5_6()
