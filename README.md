# NASA JPL Telemanom Telemetry Anomaly Detection & Statistical Profiling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: NASA Telemanom](https://img.shields.io/badge/Dataset-NASA%20SMAP%20%26%20MSL-orange.svg)](https://github.com/khundman/telemanom)

An end-to-end Exploratory Data Analysis (EDA), Statistical Analysis, Subsystem Failure Profiling, and Visual Analysis suite on the **NASA JPL Telemanom** SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory / Curiosity Rover) spacecraft anomaly detection dataset.

---

## 📌 Repository Architecture

```
ML_research/
│
├── telemetry_eda_and_statistics.ipynb             # Full End-to-End Master Notebook
├── archive/                                       # Telemanom Dataset (.npy files & labeled_anomalies.csv)
│
├── section_2_dataset_summary/
│   ├── section_2_dataset_summary.ipynb           # Dataset Architecture & Aggregation Notebook
│   ├── section_2_dataset_summary.py             # Section 2 Python Runner
│   └── README.md                                # Section 2 Documentation
│
├── section_3_statistical_profiling/
│   ├── section_3_statistical_profiling.ipynb       # Descriptive & Fault Interval Stats Notebook
│   ├── section_3_statistical_profiling.py       # Section 3 Python Runner
│   └── README.md                                # Section 3 Documentation
│
├── section_4_visual_generation/
│   ├── section_4_visual_generation.ipynb           # High-DPI Publication Figures Notebook
│   ├── section_4_visual_generation.py           # Section 4 Python Runner
│   ├── README.md                                # Section 4 Documentation
│   ├── fig1_telemetry_distributions.png         # Output Chart 1
│   ├── fig2_subsystem_anomaly_counts.png       # Output Chart 2
│   ├── fig3_anomaly_type_donut.png              # Output Chart 3
│   ├── fig4_sequence_length_violin_box.png      # Output Chart 4
│   ├── fig5_timeseries_ground_truth.png         # Output Chart 5
│   └── fig6_length_vs_duration_jointplot.png    # Output Chart 6
│
└── section_5_6_key_findings_and_summary/
    ├── section_5_6_key_findings_and_summary.ipynb # Key Findings & Verification Checklist Notebook
    ├── section_5_6_key_findings_and_summary.py # Sections 5 & 6 Python Runner
    └── README.md                                # Sections 5 & 6 Documentation
```

---

## 📊 Dataset Overview (706,971 Total Telemetry Timesteps)

| Metric Category | SMAP Mission | MSL Mission | Aggregated Combined Total |
| :--- | :--- | :--- | :--- |
| **Stream Count** | 55 streams | 27 streams | **82 telemetry streams** |
| **Train Set Rows** | 138,004 | 58,317 | **196,746 timesteps** |
| **Test Set Rows** | 435,826 | 73,729 | **510,225 timesteps** |
| **Total Telemetry Rows** | 573,830 | 132,046 | **706,971 timesteps** |
| **Feature Dimensionality** | 55 features | 25 features | **25 to 55 features** |
| **Primary Continuous Channel** | Column Index 0 (`float64`) | Column Index 0 (`float64`) | Continuous Telemetry Sensor |
| **Discrete State Features** | Columns 1–54 | Columns 1–24 | Command & Relay Vectors |

---

## 📈 Publication-Ready Figures

| Figure | Description | Output File |
| :---: | :--- | :--- |
| **Fig 1** | Primary Continuous Telemetry Distributions (SMAP normalized vs MSL raw) | `fig1_telemetry_distributions.png` |
| **Fig 2** | Subsystem Telemetry Channel & Anomaly Frequency | `fig2_subsystem_anomaly_counts.png` |
| **Fig 3** | Proportion of Anomaly Classifications (Point vs Contextual/Drift) | `fig3_anomaly_type_donut.png` |
| **Fig 4** | Sequence Length Distributions & Telemetry Variance Boxplots | `fig4_sequence_length_violin_box.png` |
| **Fig 5** | Representative Telemetry Streams with Red Ground-Truth Fault Spans | `fig5_timeseries_ground_truth.png` |
| **Fig 6** | Joint Correlation Plot (Sequence Length vs Cumulative Anomaly Duration) | `fig6_length_vs_duration_jointplot.png` |

---

## 💡 Key Findings Summary

1. **Anomaly Type Breakdown:** Contextual / Drift anomalies represent **40.95%** (43 / 105 instances), while Point anomalies represent **59.05%** (62 / 105 instances).
2. **Subsystem Vulnerability:** The **Other Avionics (A/D/G/F)** subsystem exhibits the highest failure frequency (**55 total ground-truth anomaly instances**), followed by **Actuators/Engines (E)**.
3. **Fault Duration Range:** Ground-truth anomaly durations span from a minimum of **10 timesteps** to a maximum of **4,217 timesteps**, with a median duration of **120.0 timesteps** (mean: **616.2 timesteps**).
4. **Variance Profile Impact:** Anomalous intervals show a **~6.51x average increase** in continuous sensor reading variance (**14.9714** during fault periods vs. **2.2997** during nominal steady-state periods).

---

## 🚀 Execution & Quickstart

To run the full pipeline or any specific module:

```bash
# Clone the repository
git clone https://github.com/Sourashis07/ML_research.git
cd ML_research

# Run full master notebook execution
python -m jupyter nbconvert --execute --to notebook --inplace telemetry_eda_and_statistics.ipynb

# Or run specific section notebooks
python -m jupyter nbconvert --execute --to notebook --inplace section_2_dataset_summary/section_2_dataset_summary.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace section_3_statistical_profiling/section_3_statistical_profiling.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace section_4_visual_generation/section_4_visual_generation.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace section_5_6_key_findings_and_summary/section_5_6_key_findings_and_summary.ipynb
```
