"""
Section 2: Dataset Loading & Multi-File Aggregation Pipeline
NASA JPL Telemanom Dataset Summary Module
"""

import os
import sys
import glob
import ast
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_section_2():
    print("=" * 75)
    print("      SECTION 2: DATASET LOADING & MULTI-FILE AGGREGATION PIPELINE")
    print("=" * 75)

    # Automated Path Resolution
    candidate_base_dirs = [
        'data', 'archive/data/data', 'archive/data', '.',
        '../data', '../archive/data/data', '../archive/data', '..'
    ]

    train_dir = None
    test_dir = None
    csv_path = None

    for base in candidate_base_dirs:
        tr = os.path.join(base, 'train')
        te = os.path.join(base, 'test')
        if os.path.exists(tr) and len(glob.glob(os.path.join(tr, '*.npy'))) > 0:
            train_dir = tr
            test_dir = te
            break

    for base in ['archive', 'data', '.', '../archive', '../data', '..']:
        cp = os.path.join(base, 'labeled_anomalies.csv')
        if os.path.exists(cp):
            csv_path = cp
            break

    if not train_dir or not csv_path:
        raise FileNotFoundError("Could not locate NASA Telemanom dataset directory or labeled_anomalies.csv.")

    print(f"\n[OK] Dataset Path Resolution Successful:")
    print(f"    - Training Data Directory : '{os.path.abspath(train_dir)}'")
    print(f"    - Testing Data Directory  : '{os.path.abspath(test_dir)}'")
    print(f"    - Metadata CSV Location   : '{os.path.abspath(csv_path)}'")

    # Load Metadata CSV
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
    df_meta['total_anomaly_duration'] = df_meta['anomaly_sequences'].apply(
        lambda seqs: sum(end - start for start, end in seqs) if isinstance(seqs, list) else 0
    )

    print("\n--- Labeled Anomalies Metadata Preview (First 10 Streams) ---")
    print(df_meta[['chan_id', 'spacecraft', 'num_values', 'subsystem', 'class', 'num_anomalies']].head(10).to_string(index=False))

    # Multi-File Stream Scanning
    test_file_paths = sorted(glob.glob(os.path.join(test_dir, '*.npy')))
    stream_records = []
    feature_dim_set = set()

    print("\nScanning 82 spacecraft telemetry streams...")
    for test_path in tqdm(test_file_paths, desc="Aggregating Telemetry Streams"):
        chan_id = os.path.basename(test_path).replace('.npy', '')
        train_path = os.path.join(train_dir, f"{chan_id}.npy")

        test_arr = np.load(test_path)
        train_arr = np.load(train_path) if os.path.exists(train_path) else None

        n_test_timesteps, n_features = test_arr.shape
        n_train_timesteps = train_arr.shape[0] if train_arr is not None else 0

        feature_dim_set.add(n_features)
        meta_row = df_meta[df_meta['chan_id'] == chan_id]
        spacecraft = meta_row['spacecraft'].values[0] if len(meta_row) > 0 else "Unknown"
        subsystem = meta_row['subsystem'].values[0] if len(meta_row) > 0 else "Unknown"

        stream_records.append({
            'chan_id': chan_id,
            'spacecraft': spacecraft,
            'subsystem': subsystem,
            'train_timesteps': n_train_timesteps,
            'test_timesteps': n_test_timesteps,
            'total_timesteps': n_train_timesteps + n_test_timesteps,
            'n_features': n_features
        })

    df_streams = pd.DataFrame(stream_records)

    total_train_rows = df_streams['train_timesteps'].sum()
    total_test_rows = df_streams['test_timesteps'].sum()
    total_rows = df_streams['total_timesteps'].sum()

    smap_df = df_streams[df_streams['spacecraft'] == 'SMAP']
    msl_df = df_streams[df_streams['spacecraft'] == 'MSL']

    summary_table = pd.DataFrame([
        {
            "Spacecraft": "SMAP",
            "Stream Count": len(smap_df),
            "Train Rows": f"{smap_df['train_timesteps'].sum():,}",
            "Test Rows": f"{smap_df['test_timesteps'].sum():,}",
            "Total Rows": f"{smap_df['total_timesteps'].sum():,}",
            "Feature Dimensions": f"{smap_df['n_features'].unique().tolist()}"
        },
        {
            "Spacecraft": "MSL",
            "Stream Count": len(msl_df),
            "Train Rows": f"{msl_df['train_timesteps'].sum():,}",
            "Test Rows": f"{msl_df['test_timesteps'].sum():,}",
            "Total Rows": f"{msl_df['total_timesteps'].sum():,}",
            "Feature Dimensions": f"{msl_df['n_features'].unique().tolist()}"
        },
        {
            "Spacecraft": "COMBINED TOTAL",
            "Stream Count": len(df_streams),
            "Train Rows": f"{total_train_rows:,}",
            "Test Rows": f"{total_test_rows:,}",
            "Total Rows": f"{total_rows:,}",
            "Feature Dimensions": f"{sorted(list(feature_dim_set))}"
        }
    ])

    print("\n" + "=" * 80)
    print("                     DATASET ARCHITECTURE SUMMARY TABLE")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_section_2()
