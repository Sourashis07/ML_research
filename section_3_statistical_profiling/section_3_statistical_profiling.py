"""
Section 3: Comprehensive Statistical Profiling Module
NASA JPL Telemanom Descriptive & Fault Interval Statistics
"""

import os
import sys
import glob
import ast
import re
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

def run_section_3():
    print("=" * 80)
    print("                SECTION 3: COMPREHENSIVE STATISTICAL PROFILING")
    print("=" * 80)

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

    df_meta['anomaly_sequences'] = df_meta['anomaly_sequences'].apply(parse_sequences)

    # Primary Sensor & Fault Duration Data Arrays
    smap_primary_vals = []
    msl_primary_vals = []
    nominal_variances = []
    anomalous_variances = []
    anomaly_durations = []

    print("\nProcessing telemetry streams for descriptive statistical profiling...")
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Analyzing Streams"):
        chan_id = row['chan_id']
        spacecraft = row['spacecraft']
        seqs = row['anomaly_sequences']

        test_path = os.path.join(test_dir, f"{chan_id}.npy")
        if not os.path.exists(test_path):
            continue

        arr = np.load(test_path)[:, 0] # Column 0 primary continuous sensor reading
        n_pts = len(arr)

        if spacecraft == 'SMAP':
            smap_primary_vals.append(arr)
        else:
            msl_primary_vals.append(arr)

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

    smap_concat = np.concatenate(smap_primary_vals)
    msl_concat = np.concatenate(msl_primary_vals)
    all_concat = np.concatenate([smap_concat, msl_concat])

    def compute_stats_profile(data_array, name):
        mode_res = stats.mode(data_array, keepdims=True)
        mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else np.nan
        q25, q50, q75 = np.percentile(data_array, [25, 50, 75])
        iqr = q75 - q25

        return {
            "Dataset Category": name,
            "Sample Count": len(data_array),
            "Mean": float(np.mean(data_array)),
            "Median": float(np.median(data_array)),
            "Mode": mode_val,
            "Min": float(np.min(data_array)),
            "Max": float(np.max(data_array)),
            "Range": float(np.max(data_array) - np.min(data_array)),
            "Std Dev": float(np.std(data_array)),
            "Variance": float(np.var(data_array)),
            "IQR": float(iqr),
            "Skewness": float(stats.skew(data_array)),
            "Kurtosis": float(stats.kurtosis(data_array))
        }

    df_stats_profile = pd.DataFrame([
        compute_stats_profile(all_concat, "Overall Continuous Telemetry"),
        compute_stats_profile(smap_concat, "SMAP Primary Telemetry (Normalized)"),
        compute_stats_profile(msl_concat, "MSL Primary Telemetry (Raw Unbounded)")
    ])

    print("\n" + "=" * 90)
    print("               PRIMARY CONTINUOUS SENSOR DESCRIPTIVE STATISTICS PROFILE")
    print("=" * 90)
    print(df_stats_profile.round(4).to_string(index=False))
    print("=" * 90)

    # Fault Duration & Variance Profile
    durations_arr = np.array(anomaly_durations)
    duration_stats = {
        "Total Anomaly Instances": len(durations_arr),
        "Mean Fault Duration (timesteps)": np.mean(durations_arr),
        "Median Fault Duration (timesteps)": np.median(durations_arr),
        "Min Fault Duration": np.min(durations_arr),
        "Max Fault Duration": np.max(durations_arr),
        "Std Dev of Fault Duration": np.std(durations_arr),
        "IQR of Fault Duration": np.percentile(durations_arr, 75) - np.percentile(durations_arr, 25)
    }

    variance_profile = {
        "Nominal Stream Mean Variance": np.mean(nominal_variances),
        "Nominal Stream Median Variance": np.median(nominal_variances),
        "Anomalous Stream Mean Variance": np.mean(anomalous_variances),
        "Anomalous Stream Median Variance": np.median(anomalous_variances),
        "Variance Increase Ratio (Mean)": np.mean(anomalous_variances) / (np.mean(nominal_variances) + 1e-9)
    }

    df_durations = pd.DataFrame([duration_stats]).T.reset_index()
    df_durations.columns = ["Fault Duration Metric", "Value"]

    df_variances = pd.DataFrame([variance_profile]).T.reset_index()
    df_variances.columns = ["Variance Profile Metric", "Value"]

    print("\n--- Fault Interval Duration Statistics ---")
    print(df_durations.round(2).to_string(index=False))

    print("\n--- Nominal vs. Anomalous Telemetry Variance Profile ---")
    print(df_variances.round(4).to_string(index=False))

if __name__ == "__main__":
    run_section_3()
