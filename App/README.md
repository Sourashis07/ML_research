# Autonomous Spacecraft Telemetry Anomaly Detection & 3D Orbital Simulation Platform

A production-grade, end-to-end full-stack aerospace telemetry AI platform and digital twin simulator. This platform trains an unsupervised **Spatial-Temporal Attention Autoencoder** on NASA SMAP/MSL spacecraft telemetry data, executes real-time hardware fault detection with channel-level root-cause attribution, runs a 3D WebGL low-Earth orbit (LEO) digital twin with solar flare/fault injection, and provides a dark-themed aerospace glassmorphism Mission Control Dashboard.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |      NASA SMAP / MSL Telemetry        |
                                  |     (data/train/*.npy, test/*.npy)    |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  |       Min-Max Scaler & Sliding        |
                                  |        Window Generator (W=100)       |
                                  +-------------------+-------------------+
                                                      |
                                                      v
     +------------------------------------------------+------------------------------------------------+
     |                    Spatial-Temporal Attention Autoencoder (Track A)                             |
     |                                                                                                 |
     |  +---------------------+     +-----------------------+     +---------------------------------+  |
     |  | 1D-CNN Encoder      | --> | Bidirectional GRU     | --> | Multi-Head Cross-Channel        |  |
     |  | (Spike Extraction)  |     | (Temporal Context)    |     | Self-Attention (nn.Multihead)   |  |
     |  +---------------------+     +-----------------------+     +---------------------------------+  |
     |                                                                            |                    |
     |  +-------------------------------------------------------------------------+                    |
     |  | Symmetric Decoder (Linear + GRU Head -> [B, W, C])                                           |
     |  +-------------------------------------------------------------------------------------------+  |
     +------------------------------------------------+------------------------------------------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  |      Reconstruction Residual Error    |
                                  |          e_t = |x_t - x_hat_t|        |
                                  +-------------------+-------------------+
                                                      |
                                  +-------------------+-------------------+
                                  |   Non-Parametric Dynamic Thresholding |
                                  |        (NDT & EWMA Smoothing)         |
                                  +-------------------+-------------------+
                                                      |
                                      +---------------+---------------+
                                      |                               |
                                      v                               v
                        +-------------+-------------+   +-------------+-------------+
                        | Channel Root-Cause        |   | Point-Adjusted F1 Benchmark |
                        | Attribution Breakdown     |   | Evaluation Suite            |
                        +-------------+-------------+   +-------------+-------------+
                                      |                               |
                                      +---------------+---------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  | Streamlit Mission Control HUD & 3D   |
                                  | LEO Orbit Digital Twin Simulator      |
                                  +---------------------------------------+
```

---

## 📐 Mathematical Formulation

### 1. Track A Spatial-Temporal Attention Autoencoder
The network processes continuous sensor channels $x_{t, 0}$ conditioned on command vectors $x_{t, 1..C-1}$ over sliding window $W=100$:

$$\mathbf{X}_{window} \in \mathbb{R}^{B \times W \times C}$$

**1D-CNN Temporal Feature Extraction:**
$$\mathbf{H}_{conv} = \text{GELU}(\text{BatchNorm1D}(\text{Conv1d}(\mathbf{X}_{window}^T)))$$

**Bidirectional GRU Contextual Modeling:**
$$\mathbf{H}_{gru} = \text{BiGRU}(\mathbf{H}_{conv})$$

**Multi-Head Cross-Channel Self-Attention:**
$$\mathbf{Q} = \mathbf{H}_{gru}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_{gru}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_{gru}\mathbf{W}_V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Symmetric Decoder Reconstruction:**
$$\hat{\mathbf{X}} = \text{DecoderHead}(\text{GRU}(\mathbf{H}_{attn}))$$

---

### 2. Non-Parametric Dynamic Thresholding (NDT)
Reconstruction residual error magnitude:
$$e_t = |x_t - \hat{x}_t|$$

EWMA Smoothing over window:
$$e_t^{(smooth)} = \alpha e_t + (1 - \alpha) e_{t-1}^{(smooth)}$$

Dynamic Threshold Bound:
$$\epsilon_{thresh} = \mu(e_t^{(smooth)}) + 3 \cdot \sigma(e_t^{(smooth)})$$

---

### 3. Channel-Level Root-Cause Attribution
Normalized attribution score for each subsystem channel $c \in \{1, \dots, C\}$ at timestep $t$:

$$\text{Contribution}(c) = \frac{|x_{t, c} - \hat{x}_{t, c}|}{\sum_{j=1}^{C} |x_{t, j} - \hat{x}_{t, j}|}$$

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.9+
- PyTorch 2.0+

### Installation

```bash
# Navigate to App directory
cd App

# Install requirements
pip install -r requirements.txt
```

### Running Automated Test Suite
Verify model forward pass, NDT threshold calculation, root-cause attribution, and point-adjusted $F_1$ metrics:

```bash
pytest tests/
```

### Running Synthetic Telemetry Dataset Generator
If raw NASA dataset is not present, generate realistic synthetic SMAP ($55$ channels) and MSL ($25$ channels) arrays:

```bash
python scripts/generate_synthetic_telemetry.py
```

### Launching Mission Control Web Application
Start the Streamlit dark-themed aerospace HUD and 3D digital twin:

```bash
streamlit run app.py
```

---

## 🔬 NASA Telemanom Research Mapping
- **Original Paper:** *Detecting Spacecraft Anomalies Using LSTMs and Non-Parametric Dynamic Thresholding* (Hundman et al., KDD 2018).
- **Benchmark Datasets:**
  - **SMAP:** Soil Moisture Active Passive satellite telemetry.
  - **MSL:** Mars Science Laboratory (Curiosity Rover) telemetry.
- **Model Enhancement:** This platform upgrades standard LSTM architectures to a **Spatial-Temporal Attention Autoencoder** combining 1D-CNN transient extraction, BiGRU sequence modeling, and Multi-Head Cross-Attention.
