# Section 3: Comprehensive Statistical Profiling Module

This module contains the executable Jupyter Notebook [`section_3_statistical_profiling.ipynb`](section_3_statistical_profiling.ipynb) and Python script [`section_3_statistical_profiling.py`](section_3_statistical_profiling.py) for computing descriptive statistical profiles (Central Tendency, Dispersion, Extremes, Skewness, Kurtosis) across normalized SMAP and raw MSL continuous telemetry channels, as well as ground-truth fault interval durations and stream variance profiles.

---

## 📋 What This Module Does

1. **Central Tendency Analysis:** Computes Mean, Median, and Mode (via `scipy.stats.mode`) for overall, SMAP, and MSL telemetry channels.
2. **Dispersion & Extremes:** Computes Minimum, Maximum, Total Range ($\text{Max} - \text{Min}$), Standard Deviation ($\sigma$), Variance ($\sigma^2$), and Interquartile Range (IQR = $Q_3 - Q_1$).
3. **Distribution Shape Analysis:** Evaluates Skewness (asymmetry) and Kurtosis (tail heaviness / peaking).
4. **Fault Duration Profiling:** Analyzes the distribution of anomalous sequence lengths (Min, Max, Mean, Median, Std Dev, and IQR in timesteps).
5. **Nominal vs. Anomalous Variance Comparison:** Calculates signal volatility differences during nominal steady-state operation versus ground-truth anomaly windows (~6.51x variance increase during faults).

---

## 🛠️ Files Included

- **`section_3_statistical_profiling.ipynb`**: Interactive Jupyter Notebook with formatted markdown explanations and executed statistical summary tables.
- **`section_3_statistical_profiling.py`**: Standalone Python script for terminal execution.

---

## 🚀 How to Run

### Option 1: Open and Run Jupyter Notebook
Open `section_3_statistical_profiling.ipynb` in VS Code / Jupyter Lab and run all cells.

### Option 2: Run via Terminal
```bash
python -m jupyter nbconvert --execute --to notebook --inplace section_3_statistical_profiling/section_3_statistical_profiling.ipynb
```
or
```bash
python section_3_statistical_profiling/section_3_statistical_profiling.py
```
