import os
import numpy as np
import pandas as pd
import json

def generate_synthetic_channel(chan_id, spacecraft="SMAP", num_timesteps=8500, num_channels=55, seed=42):
    """
    Generates a realistic continuous + binary command telemetry array matching JPL Telemanom format.
    - Col 0: Main continuous sensor channel (e.g. voltage, temperature, current)
    - Col 1..num_channels-1: Binary operational command vectors (0 or 1)
    """
    np.random.seed(seed + hash(chan_id) % 10000)
    
    t = np.linspace(0, 100, num_timesteps)
    # Baseline nominal signal: orbital sine wave + high frequency noise
    base_signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 15.0) + 0.05 * np.cos(2 * np.pi * t / 3.2)
    noise = np.random.normal(0, 0.02, size=num_timesteps)
    col0 = base_signal + noise

    # Create binary command channels (Cols 1+)
    command_cols = []
    for c in range(1, num_channels):
        freq = 1.0 + (c % 5) * 2.5
        cmd_trigger = (np.sin(2 * np.pi * t / freq) > 0.4).astype(float)
        command_cols.append(cmd_trigger)

    # Combine into array [N, C]
    telemetry = np.column_stack([col0] + command_cols)
    return telemetry

def create_synthetic_dataset(output_dir="App/data", seed=42):
    """
    Creates synthetic train and test .npy files + labeled_anomalies.csv
    matching exact SMAP (55 channels) and MSL (25 channels) specifications.
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    channels = [
        {"chan_id": "P-1", "spacecraft": "SMAP", "num_channels": 55, "anomalies": [[2149, 2349], [4536, 4844], [3539, 3779]], "classes": ["contextual", "contextual", "contextual"]},
        {"chan_id": "S-1", "spacecraft": "SMAP", "num_channels": 55, "anomalies": [[5300, 5747]], "classes": ["point"]},
        {"chan_id": "E-1", "spacecraft": "SMAP", "num_channels": 55, "anomalies": [[5000, 5030], [5610, 6086]], "classes": ["contextual", "contextual"]},
        {"chan_id": "M-1", "spacecraft": "MSL",  "num_channels": 25, "anomalies": [[4000, 4300]], "classes": ["point"]},
        {"chan_id": "M-2", "spacecraft": "MSL",  "num_channels": 25, "anomalies": [[5200, 5600]], "classes": ["contextual"]},
    ]

    anomaly_rows = []

    for idx, item in enumerate(channels):
        cid = item["chan_id"]
        sc = item["spacecraft"]
        num_c = item["num_channels"]
        num_steps = 8500

        # Generate nominal train set
        train_data = generate_synthetic_channel(cid, spacecraft=sc, num_timesteps=6000, num_channels=num_c, seed=seed+idx)
        np.save(os.path.join(train_dir, f"{cid}.npy"), train_data)

        # Generate test set with injected anomalies
        test_data = generate_synthetic_channel(cid, spacecraft=sc, num_timesteps=num_steps, num_channels=num_c, seed=seed+idx+100)

        # Inject anomalies into test set
        for seq in item["anomalies"]:
            start, end = seq
            # Inject spike, drop, or flatline anomaly in col 0 and related command cols
            anomaly_type = np.random.choice(["spike", "drop", "drift", "flatline"])
            if anomaly_type == "spike":
                test_data[start:end, 0] += np.random.uniform(0.4, 0.8, size=(end-start))
                test_data[start:end, 1:4] += 0.5
            elif anomaly_type == "drop":
                test_data[start:end, 0] -= np.random.uniform(0.3, 0.6, size=(end-start))
            elif anomaly_type == "drift":
                drift = np.linspace(0, 0.7, end - start)
                test_data[start:end, 0] += drift
            else:
                test_data[start:end, 0] = test_data[start, 0]

        np.save(os.path.join(test_dir, f"{cid}.npy"), test_data)

        anomaly_rows.append({
            "chan_id": cid,
            "spacecraft": sc,
            "anomaly_sequences": json.dumps(item["anomalies"]),
            "class": json.dumps(item["classes"]),
            "num_values": num_steps
        })

    df = pd.DataFrame(anomaly_rows)
    df.to_csv(os.path.join(output_dir, "labeled_anomalies.csv"), index=False)
    print(f"[+] Successfully generated synthetic telemetry dataset in '{output_dir}' ({len(channels)} channels).")

if __name__ == "__main__":
    create_synthetic_dataset(output_dir="App/data")
