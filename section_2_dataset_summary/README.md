# Section 2: Dataset Loading & Multi-File Aggregation Pipeline

This module contains the executable Jupyter Notebook [`section_2_dataset_summary.ipynb`](section_2_dataset_summary.ipynb) and Python script [`section_2_dataset_summary.py`](section_2_dataset_summary.py) for automated path resolution, metadata parsing, multi-file telemetry stream scanning, and outputting formatted dataset summary statistics for the NASA JPL Telemanom SMAP and MSL anomaly detection dataset.

---

## 📋 What This Module Does

1. **Automated Path Resolution:** Locates train/test `.npy` directories and `labeled_anomalies.csv` across multiple fallback candidate paths (`./data`, `./archive/data/data`, `../archive`, etc.).
2. **Metadata Parsing:** Loads `labeled_anomalies.csv` and parses stringified Python/JSON arrays (`anomaly_sequences` and `class`) using `ast.literal_eval` and regex tokenization.
3. **Subsystem Mapping:** Maps spacecraft channel IDs to operational subsystem categories:
   - `Power (P)`: Power management telemetry
   - `Thermal/Radiation (T/S)`: Temperature and radiation sensors
   - `Actuators/Engines (E)`: Propulsion and engine telemetry
   - `Other Avionics (A/D/G/F/M/C/B/R)`: Flight control, guidance, and payload instruments
4. **Multi-File Aggregation:** Iterates through 82 spacecraft telemetry streams (55 SMAP, 27 MSL) to inspect sequence lengths, primary continuous sensor channels (Column index 0), and discrete command vectors (Columns 1+).
5. **Formatted Summary Output:** Computes and prints row counts across train/test splits, total aggregated rows (706,971 timesteps), and feature dimensions (25 features for MSL, 55 features for SMAP).

---

## 🛠️ Files Included

- **`section_2_dataset_summary.ipynb`**: Interactive Jupyter Notebook with rich Markdown headers, executed cell outputs, and rendered summary DataFrames.
- **`section_2_dataset_summary.py`**: Standalone Python script for terminal execution.

---

## 🚀 How to Run

### Option 1: Open and Run Jupyter Notebook
Open `section_2_dataset_summary.ipynb` in VS Code / Jupyter Lab and run all cells.

### Option 2: Run via Terminal
```bash
python -m jupyter nbconvert --execute --to notebook --inplace section_2_dataset_summary/section_2_dataset_summary.ipynb
```
or
```bash
python section_2_dataset_summary/section_2_dataset_summary.py
```
