# Section 4: Visual Generation Suite Module

This module contains the executable Jupyter Notebook [`section_4_visual_generation.ipynb`](section_4_visual_generation.ipynb) and Python script [`section_4_visual_generation.py`](section_4_visual_generation.py) for configuring publication-ready plot aesthetics (`sns.set_theme(style='whitegrid')`, 300 DPI high resolution) and generating 6 distinct visual charts saved as high-resolution PNG files.

---

## 📋 Generated Figure Specifications

1. **`fig1_telemetry_distributions.png`**: Multi-panel distribution plots comparing SMAP continuous sensor values (normalized $[-1, 1]$) with MSL raw unbounded continuous readings.
2. **`fig2_subsystem_anomaly_counts.png`**: Grouped bar chart comparing the total count of unique telemetry channels and ground-truth anomaly instances grouped by subsystem category (`Power`, `Thermal/Radiation`, `Actuators/Engines`, `Other Avionics`).
3. **`fig3_anomaly_type_donut.png`**: Donut chart displaying the proportion of Point Anomalies (59.05%) vs. Contextual / Drift Anomalies (40.95%).
4. **`fig4_sequence_length_violin_box.png`**: Violin plot of sequence lengths (`num_values`) and log-scale Boxplot of primary sensor variance across SMAP vs MSL.
5. **`fig5_timeseries_ground_truth.png`**: 3-panel representative time series line chart (`P-1`, `T-1`, `E-1`), explicitly shading ground-truth fault intervals in red (`ax.axvspan`).
6. **`fig6_length_vs_duration_jointplot.png`**: Joint scatter plot with marginal distributions illustrating Pearson correlation ($r = 0.448$) between sequence length and total anomaly duration per channel.

---

## 🛠️ Files Included

- **`section_4_visual_generation.ipynb`**: Interactive Jupyter Notebook with inline figure displays, cell executions, and markdown section headers.
- **`section_4_visual_generation.py`**: Standalone Python script for generating and exporting figures.
- **Saved PNG Files**: `fig1_telemetry_distributions.png` through `fig6_length_vs_duration_jointplot.png`.

---

## 🚀 How to Run

### Option 1: Open and Run Jupyter Notebook
Open `section_4_visual_generation.ipynb` in VS Code / Jupyter Lab and run all cells.

### Option 2: Run via Terminal
```bash
python -m jupyter nbconvert --execute --to notebook --inplace section_4_visual_generation/section_4_visual_generation.ipynb
```
or
```bash
python section_4_visual_generation/section_4_visual_generation.py
```
