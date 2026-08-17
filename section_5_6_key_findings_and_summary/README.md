# Sections 5 & 6: Key Findings & Summary Module

This module contains the executable Jupyter Notebook [`section_5_6_key_findings_and_summary.ipynb`](section_5_6_key_findings_and_summary.ipynb) and Python script [`section_5_6_key_findings_and_summary.py`](section_5_6_key_findings_and_summary.py) for programmatically synthesizing data-driven key findings, printing a software verification checklist, and documenting operational telemetry modeling recommendations.

---

## 📋 What This Module Does

1. **Automated Key Findings Generator (Section 5):**
   - **Bullet 1 [Anomaly Type Breakdown]:** Contextual/Drift anomalies represent **40.95%** (43 / 105 instances), while Point anomalies represent **59.05%** (62 / 105 instances).
   - **Bullet 2 [Subsystem Vulnerability]:** The **Other Avionics (A/D/G/F)** subsystem exhibits the highest failure frequency (**44 total ground-truth instances**), followed by **Actuators/Engines (E)** with **21 instances**.
   - **Bullet 3 [Fault Duration Range]:** Fault durations range from **10 timesteps** to **4,217 timesteps**, with a median duration of **120.0 timesteps** (mean: **616.2 timesteps**).
   - **Bullet 4 [Variance Profile Impact]:** Anomalous intervals show a **~6.51x mean increase** in continuous sensor reading variance (**14.9714** during faults vs **2.2997** during nominal state).

2. **Software Environment Verification Checklist (Section 6):**
   - Explicitly confirms software versions for `NumPy`, `Pandas`, `SciPy`, `Matplotlib`, `Seaborn`, and `Python`.

3. **Operational Recommendations:**
   - Guidance on telemetry normalization, feature scaling, subsystem anomaly prioritization, and threshold selection (e.g., Telemanom EWMA rolling residual thresholding).

---

## 🛠️ Files Included

- **`section_5_6_key_findings_and_summary.ipynb`**: Interactive Jupyter Notebook displaying formatted summary DataFrames, automated finding bullet points, and software checklists.
- **`section_5_6_key_findings_and_summary.py`**: Standalone Python script for terminal execution.

---

## 🚀 How to Run

### Option 1: Open and Run Jupyter Notebook
Open `section_5_6_key_findings_and_summary.ipynb` in VS Code / Jupyter Lab and run all cells.

### Option 2: Run via Terminal
```bash
python -m jupyter nbconvert --execute --to notebook --inplace section_5_6_key_findings_and_summary/section_5_6_key_findings_and_summary.ipynb
```
or
```bash
python section_5_6_key_findings_and_summary/section_5_6_key_findings_and_summary.py
```
