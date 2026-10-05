import os
import sys
import shutil
import pandas as pd

sys.path.append(os.path.dirname(__file__))

try:
    from scripts.generate_synthetic_telemetry import create_synthetic_dataset
except ImportError:
    try:
        from generate_synthetic_telemetry import create_synthetic_dataset
    except ImportError:
        from .generate_synthetic_telemetry import create_synthetic_dataset

def prepare_data(app_data_dir="App/data", archive_dir="archive"):
    """
    Sets up the dataset directory inside App/data.
    1. Checks if archive/ contains real dataset.
    2. Copies available .npy files and labeled_anomalies.csv.
    3. If archive is missing or empty, triggers synthetic generation.
    """
    train_dest = os.path.join(app_data_dir, "train")
    test_dest = os.path.join(app_data_dir, "test")
    csv_dest = os.path.join(app_data_dir, "labeled_anomalies.csv")

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    archive_train = os.path.join(archive_dir, "data", "data", "train")
    archive_test = os.path.join(archive_dir, "data", "data", "test")
    archive_csv = os.path.join(archive_dir, "labeled_anomalies.csv")

    has_real_data = False

    if os.path.exists(archive_train) and os.path.exists(archive_test):
        train_files = [f for f in os.listdir(archive_train) if f.endswith('.npy')]
        test_files = [f for f in os.listdir(archive_test) if f.endswith('.npy')]

        if len(train_files) > 0 and len(test_files) > 0:
            print(f"[+] Found real JPL dataset in '{archive_dir}'. Copying to '{app_data_dir}'...")
            for tf in train_files:
                shutil.copy2(os.path.join(archive_train, tf), os.path.join(train_dest, tf))
            for tf in test_files:
                shutil.copy2(os.path.join(archive_test, tf), os.path.join(test_dest, tf))
            if os.path.exists(archive_csv):
                shutil.copy2(archive_csv, csv_dest)
            has_real_data = True

    if not has_real_data:
        # Check if App/data already has valid .npy files
        existing_train = [f for f in os.listdir(train_dest) if f.endswith('.npy')]
        if len(existing_train) == 0:
            print("[!] No raw data found in archive. Generating synthetic NASA SMAP/MSL dataset...")
            create_synthetic_dataset(output_dir=app_data_dir)
        else:
            print(f"[+] App/data already populated with {len(existing_train)} files.")

    print(f"[+] Data preparation complete. Train files: {len(os.listdir(train_dest))}, Test files: {len(os.listdir(test_dest))}")

if __name__ == "__main__":
    # Resolve relative paths from current working directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_data = os.path.join(base_dir, "data")
    archive_path = os.path.abspath("archive")
    prepare_data(app_data_dir=app_data, archive_dir=archive_path)
