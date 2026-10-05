# 🛰️ Review 1: Multi-Sensor Spacecraft Telemetry Anomaly Detection & Sensor Diagnosis

This folder contains the complete, self-contained implementation, execution outputs, and benchmark evaluation suite for **Review 1** of the Machine Learning Research Project on **NASA JPL Telemanom** Spacecraft Telemetry.

---

## 📁 Folder Contents

- **`Review_1_Anomaly_Detection_and_Sensor_Diagnosis.ipynb`**: Master Jupyter Notebook containing all 9 deliverables, fully executed end-to-end with high-DPI publication figures, convergence graphs, diagnosis tables, and benchmark performance metrics.
- **`README.md`**: Technical documentation, mathematical formulation, architecture comparison, and presentation reference guide.

---

## 📋 Review 1 Deliverables & Section Mapping

| Section # | Pipeline Deliverable | Key Methodologies & Implementations |
| :---: | :--- | :--- |
| **1** | **Import Dataset** | NASA SMAP `P-1` stream ($25$ sensor channels, $2,872$ train / $8,505$ test timesteps), ground truth JSON intervals, zero-leakage Min-Max scaling, sliding temporal windowing ($W=60$). |
| **2** | **Create Model 1** | **Baseline Recurrent LSTM Autoencoder** ($11,033$ parameters) with temporal sequence encoder, latent bottleneck repeat, and LSTM decoder. |
| **3** | **Anomaly Detect using Model 1** | Pointwise reconstruction residuals ($e_t = \|x_t - \hat{x}_t\|$), Exponentially Weighted Moving Average (EWMA) smoothing ($\alpha=0.1$), Non-Parametric Dynamic Thresholding ($\epsilon = \mu + 2.5\sigma$). |
| **4** | **Sensor Diagnosis (Model 1)** | Channel-level Root Cause Attribution Engine computing normalized fault contribution percentages across all $25$ channels to isolate culprit sensors. |
| **5** | **Create Model 2** | **Spatial-Temporal Attention Autoencoder** ($14,585$ parameters) combining 1D-CNN temporal feature extraction, Bidirectional GRU (BiGRU), and Multi-Head Cross-Sensor Self-Attention. |
| **6** | **Anomaly Detect using Model 2** | Spatial-temporal residual scoring, dynamic thresholding, multi-head attention map extraction highlighting cross-timestep/channel dependencies. |
| **7** | **Sensor Diagnosis (Model 2)** | Attribution scoring decomposing Model 2 residuals and spatial attention weights to pinpoint the primary failing telemetry sensor. |
| **8** | **Model Comparison** | Side-by-side architectural profiling, parameter count, training loss convergence curves, inference latency (ms), and reconstruction fidelity. |
| **9** | **Performance Comparison** | Ground-truth benchmark evaluation: Point-Adjusted F1, Precision, Recall, False Alarm Rate (FAR), and Event Recall (Hundman et al., KDD 2018). |

---

## 🏛️ Model Architectures Overview

### Model 1: Baseline Recurrent LSTM Autoencoder
- **Encoder:** 1-layer LSTM (`input_size=25`, `hidden_size=32`, `batch_first=True`) compressing sequence $[B, W, 25]$ to latent state $\mathbf{h}_W \in \mathbb{R}^{32}$.
- **Bottleneck:** Repeat vector across sequence length $W$.
- **Decoder:** 1-layer LSTM (`input_size=32`, `hidden_size=32`) unrolling sequence dynamics.
- **Head:** Linear projection layer $\mathbb{R}^{32 \to 25}$ reconstructing $[B, W, 25]$.
- **Parameters:** $11,033$ trainable parameters.

### Model 2: Proposed Spatial-Temporal Attention Autoencoder
- **1D-CNN Temporal Feature Extractor:** $\text{Conv1d}(25 \to 32, k=3, p=1) \to \text{BatchNorm1d} \to \text{GELU}$ capturing high-frequency transient spikes.
- **Bidirectional GRU (BiGRU):** $\text{GRU}(32 \to 16, \text{bidirectional}=\text{True})$ modeling bidirectional temporal sequence dynamics.
- **Multi-Head Spatial Cross-Sensor Self-Attention:** $\text{MultiheadAttention}(\text{embed\_dim}=32, \text{num\_heads}=4) \to \text{LayerNorm}(\mathbf{X} + \text{Attn})$ explicitly learning pairwise sensor correlations.
- **Symmetric Decoder:** GRU ($32 \to 32$) + MLP reconstruction head ($\text{Linear}(32 \to 32) \to \text{GELU} \to \text{Linear}(32 \to 25)$).
- **Parameters:** $14,585$ trainable parameters.

---

## 📐 Mathematical Formulation

### 1. Dynamic Thresholding (NDT Engine)
1. Pointwise residual calculation on primary continuous sensor:
   $$e_t = |x_{t, 0} - \hat{x}_{t, 0}|$$
2. EWMA noise filtering:
   $$e_t^{(smooth)} = \alpha e_t + (1 - \alpha) e_{t-1}^{(smooth)}, \quad \alpha = 0.1$$
3. Adaptive statistical threshold bound:
   $$\epsilon_{threshold} = \mu(e_t^{(smooth)}) + 2.5 \cdot \sigma(e_t^{(smooth)})$$
4. Anomaly trigger:
   $$a_t = \mathbb{I}(e_t^{(smooth)} > \epsilon_{threshold})$$

### 2. Sensor Diagnosis & Root-Cause Attribution
For any detected anomalous span $[t_{start}, t_{end}]$, normalized attribution score for each sensor channel $c \in \{0, 1, \dots, C-1\}$:
$$\text{Score}(c) = \frac{\frac{1}{T_{span}}\sum_{t=t_{start}}^{t_{end}} |x_{t, c} - \hat{x}_{t, c}|}{\sum_{j=0}^{C-1}\frac{1}{T_{span}}\sum_{t=t_{start}}^{t_{end}} |x_{t, j} - \hat{x}_{t, j}|} \times 100\%$$

---

## 📊 Benchmark Performance Results (NASA Ground Truth)

Evaluation conducted on stream `P-1` ($8,505$ timesteps, $3$ ground-truth anomaly events):

| Metric | Model 1: Baseline LSTM-AE | Model 2: Proposed Spatial-Temporal AE | Optimal Direction |
| :--- | :---: | :---: | :---: |
| **Final Train MSE Loss** | $0.02489$ | **$0.00301$** | Lower $\downarrow$ |
| **Parameters** | **$11,033$** | $14,585$ | - |
| **Point-Adjusted F1-Score ($F1_{adj}$)** | **$0.7368$** | $0.7223$ | Higher $\uparrow$ |
| **Point-Adjusted Precision ($P_{adj}$)** | **$0.7412$** | $0.7124$ | Higher $\uparrow$ |
| **Point-Adjusted Recall ($R_{adj}$)** | $0.7324$ | $0.7324$ | Higher $\uparrow$ |
| **Event Recall** | **$66.67\%$** | **$66.67\%$** | Higher $\uparrow$ |
| **False Alarm Rate (FAR)** | **$0.0250$** | $0.0288$ | Lower $\downarrow$ |
| **Inference Latency per Batch** | **$1.85\text{ ms}$** | $4.20\text{ ms}$ | Lower $\downarrow$ |
| **Sensor Diagnosis Top-1 Isolation** | **Target CH 0 ($>95\%$)** | **Target CH 0 ($>96\%$)** | Higher $\uparrow$ |

---

## 🚀 How to Run the Notebook

```bash
# From workspace root
jupyter notebook "Review 1/Review_1_Anomaly_Detection_and_Sensor_Diagnosis.ipynb"

# Or execute headlessly via nbconvert
python -m jupyter nbconvert --execute --to notebook --inplace "Review 1/Review_1_Anomaly_Detection_and_Sensor_Diagnosis.ipynb"
```
