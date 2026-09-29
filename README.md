# ⚡ ChronosDeep: Deep Learning Time Series Forecasting Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

An end-to-end, production-grade deep learning time series demand forecasting engine. ChronosDeep integrates four deep neural architectures—**Multilayer Perceptron (MLP)**, **1D Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)**, and a **Hybrid CNN-LSTM Spatiotemporal Network**—with a high-performance **FastAPI asynchronous backend** and an interactive, real-time **glassmorphism analytical dashboard**.

---

## 📌 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Mathematical & Theoretical Formulation](#-mathematical--theoretical-formulation)
3. [Deep Learning Model Zoo](#-deep-learning-model-zoo)
4. [Empirical Benchmark Results](#-empirical-benchmark-results)
5. [API Specification & Endpoints](#-api-specification--endpoints)
6. [Interactive Web Dashboard](#-interactive-web-dashboard)
7. [Repository Structure](#-repository-structure)
8. [Local Installation & Setup](#-local-installation--setup)
9. [Production Cloud Deployment](#-production-cloud-deployment)
10. [License & Acknowledgements](#-license--acknowledgements)

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Kaggle Store Item Data  │
                                  │      (5+ Year Dataset)    │
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │   Sliding-Window Pipeline │
                                  │   (Window=30, Lag=90)     │
                                  └─────────────┬─────────────┘
                                                │
               ┌────────────────┬───────────────┴───────────────┬────────────────┐
               │                │                               │                │
        ┌──────▼──────┐  ┌──────▼──────┐                 ┌──────▼──────┐  ┌──────▼──────┐
        │  MLP Dense  │  │   1D CNN    │                 │    LSTM     │  │   CNN-LSTM  │
        │  Baseline   │  │ Feature Ext │                 │  Recurrent  │  │    Hybrid   │
        └──────┬──────┘  └──────┬──────┘                 └──────┬──────┘  └──────┬──────┘
               │                │                               │                │
               └────────────────┴───────────────┬───────────────┴────────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │   Serialized .keras Cache │
                                  │  + Validation Metrics JSON│
                                  └─────────────┬─────────────┘
                                                │
                        ┌───────────────────────▼───────────────────────┐
                        │       FastAPI Production Async Engine         │
                        │   • In-Memory Dataset & Tensor Caching        │
                        │   • Multi-Step Rolling Autoregression         │
                        │   • Custom CSV Parsing & Streaming Pipeline   │
                        └───────────────┬───────────────────────┬───────┘
                                        │                       │
               ┌────────────────────────▼────────┐     ┌────────▼────────────────────────┐
               │  REST API Endpoints (/api/...)  │     │   Glassmorphism Web Dashboard   │
               │  • /api/forecast  • /api/compare│     │   • Real-Time Chart.js Visuals  │
               │  • /api/history   • /api/health │     │   • Multi-Model Overlays        │
               │  • /api/metadata  • /api/upload │     │   • Custom Sequence Projection  │
               └─────────────────────────────────┘     └─────────────────────────────────┘
```

---

## 📐 Mathematical & Theoretical Formulation

### 1. Problem Definition
Given a sequence of historical sales observations $Y = \{y_1, y_2, \dots, y_T\}$, the objective is to predict future values across a multi-step forecast horizon $H = \{y_{T+1}, y_{T+2}, \dots, y_{T+h}\}$ where $h \in [7, 90]$.

### 2. Supervised Sliding Window Transformation
The raw univariate time series is transformed into supervised pairs $(X, y)$ using a fixed lookback window $W = 30$:
$$X_t = [y_{t-W+1}, y_{t-W+2}, \dots, y_t] \in \mathbb{R}^{W}$$
$$y_{t+1} = y_{t+1} \in \mathbb{R}$$

### 3. Autoregressive Rolling Multi-Step Forecasting
Multi-step projections are computed iteratively using rolling autoregression:
$$\hat{y}_{T+1} = f_\theta([y_{T-W+1}, \dots, y_T])$$
$$\hat{y}_{T+2} = f_\theta([y_{T-W+2}, \dots, y_T, \hat{y}_{T+1}])$$
$$\hat{y}_{T+k} = f_\theta([\dots, \hat{y}_{T+k-1}])$$

Where non-negativity is enforced as a domain constraint: $\hat{y}_{T+k} \leftarrow \max(0, \hat{y}_{T+k})$.

---

## 🧠 Deep Learning Model Zoo

| Architecture | Input Tensor Shape | Internal Layer Hierarchy | Strengths & Computational Profile |
| :--- | :---: | :--- | :--- |
| **MLP (Multilayer Perceptron)** | `(batch, 30)` | • Dense(100, ReLU)<br>• Dense(1, Linear) | Ultra-fast feedforward baseline; captures static cross-timestep correlations without temporal ordering assumptions. |
| **1D CNN** | `(batch, 30, 1)` | • Conv1D(64, kernel=2, ReLU)<br>• MaxPooling1D(2)<br>• Flatten → Dense(50, ReLU)<br>• Dense(1, Linear) | Invariant local feature extraction; captures rapid trend changes and sub-period wave patterns via 1D convolutions. |
| **LSTM** | `(batch, 30, 1)` | • LSTM(50, ReLU)<br>• Dense(1, Linear) | Explicit sequence memory mechanism; handles short and medium term temporal dependencies through gating units. |
| **CNN-LSTM Hybrid** | `(batch, 2, 15, 1)` | • TimeDistributed(Conv1D(64))<br>• TimeDistributed(MaxPooling1D)<br>• TimeDistributed(Flatten)<br>• LSTM(50, ReLU)<br>• Dense(1, Linear) | Spatiotemporal feature representation; CNN encodes local 15-day sub-windows while the LSTM models inter-window trajectory dynamics. |

### LSTM Gating Equations
The LSTM recurrent cell computes internal state transitions using input gate $i_t$, forget gate $f_t$, cell candidate $\tilde{C}_t$, and output gate $o_t$:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

---

## 📊 Empirical Benchmark Results

Models were evaluated on out-of-sample store sales validation splits with identical training budgets (12 epochs, batch size 512, Adam optimizer, early stopping on validation loss):

| Model Architecture | Train Loss (MSE) | Validation Loss (MSE) | Validation RMSE | Inference Latency (Batch 1) | Model File Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP Baseline** | 337.09 | 342.25 | **18.50** | ~1.2 ms | 59.4 KB |
| **1D CNN** | 346.70 | 351.94 | **18.76** | ~2.4 ms | 571.3 KB |
| **LSTM Recurrent** | 399.20 | 351.94 | **18.76** | ~4.1 ms | 149.7 KB |
| **CNN-LSTM Hybrid** | 368.64 | 367.49 | **19.17** | ~5.8 ms | 1.24 MB |

---

## 🔌 API Specification & Endpoints

The backend is exposed via a production-ready **FastAPI** application with auto-generated OpenAPI / Swagger UI available at `/docs`.

### Key Endpoints

#### 1. `GET /api/health`
Health check for cloud orchestrators (Render, Kubernetes, AWS ALB).
```json
{
  "status": "healthy",
  "service": "Deep Learning Time Series Forecasting",
  "version": "1.0.0"
}
```

#### 2. `GET /api/metadata`
Returns metadata on available stores (1–10), items (1–50), and neural model scorecards.

#### 3. `GET /api/history?store_id=1&item_id=1&days=90`
Fetches historical daily sales sequence for specified retail store and SKU.

#### 4. `POST /api/forecast`
Executes rolling neural inference for a single selected architecture.
```json
// Request Body
{
  "store_id": 1,
  "item_id": 1,
  "model_name": "LSTM",
  "horizon": 30
}
```
```json
// Response Body
{
  "model": "LSTM",
  "store_id": 1,
  "item_id": 1,
  "horizon": 30,
  "forecast": [
    { "date": "2018-01-01", "sales": 26.19 },
    { "date": "2018-01-02", "sales": 27.57 }
  ],
  "summary": {
    "total_projected_sales": 874.3,
    "avg_daily_sales": 29.14,
    "peak_day": "2018-01-21"
  }
}
```

#### 5. `POST /api/compare`
Runs concurrent inference across all 4 architectures on identical historical inputs for side-by-side benchmarking.

#### 6. `POST /api/upload-csv`
Accepts multipart file upload (`.csv` format containing `sales` column) and runs instantaneous neural forecasting.

---

## 💻 Interactive Web Dashboard

The frontend is engineered with zero runtime build dependencies for maximum speed and visual fidelity:
* **Glassmorphism Dark UI**: Built with custom CSS custom properties, backdrop filters, and subtle ambient gradients.
* **Dynamic Chart.js Engine**: Dual-curve charting (solid line for historical context, dashed glowing curves for neural projections).
* **Multi-Model Benchmark Layer**: Side-by-side visualization comparing trajectory variations between Conv1D, LSTM, and Hybrid models.
* **Instant Horizon Slider**: Real-time forecast recalculation from 7 to 90 days.
* **Custom Dataset Upload Modal**: Drag-and-drop CSV ingestion with automatic column inference.

---

## 📁 Repository Structure

```
Time_Series_Forecasting/
├── backend/
│   ├── app.py                      # FastAPI REST API & lifespan caching
│   ├── inference.py                # Preprocessing & autoregressive neural engine
│   └── train_export_models.py      # Automated model training & weight serialization
├── saved_models/
│   ├── model_mlp.keras             # MLP model weights
│   ├── model_cnn.keras             # 1D CNN model weights
│   ├── model_lstm.keras            # LSTM model weights
│   ├── model_cnn_lstm.keras        # CNN-LSTM hybrid weights
│   └── metrics.json                # Model benchmark metadata
├── static/
│   ├── css/
│   │   └── styles.css              # Glassmorphism dark theme & typography
│   ├── js/
│   │   └── app.js                  # Dynamic Chart.js logic & API binding
│   └── index.html                  # Semantic HTML5 dashboard layout
├── output/                         # Visualization charts & preview assets
│   ├── overview.gif                # Dashboard demo animation
│   ├── compare models.PNG          # Model comparison plot
│   ├── store sales.PNG             # Store sales EDA plot
│   ├── overall daily sales.PNG     # Aggregated trend plot
│   └── item daily sales.PNG        # Item distribution plot
├── Dockerfile                      # Production container image configuration
├── render.yaml                     # Render 1-click cloud blueprint
├── requirements.txt                # Pinned Python dependencies
├── train.csv                       # Historical daily sales dataset
├── test.csv                        # Evaluation dataset
├── DEPLOYMENT_GUIDE.md             # Complete step-by-step deployment guide
└── README.md                       # Technical documentation
```

---

## 🛠️ Local Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Step-by-Step Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sAkhil2027/Time_Series_Forecasting.git
   cd Time_Series_Forecasting
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Re-train Neural Models:**
   ```bash
   python backend/train_export_models.py
   ```

5. **Start the FastAPI Server:**
   ```bash
   uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
   ```

6. **Access Dashboard & Interactive Docs:**
   * **Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
   * **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * **Redoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🚀 Production Cloud Deployment

### 1. Render.com (1-Click Deployment)
This repository includes a [`render.yaml`](render.yaml) blueprint:
1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com/) and click **"New" → "Blueprint"**.
3. Connect your repository; Render will configure and deploy the service automatically.

### 2. Docker Container Deployment
Build and run the containerized application locally or on any cloud host:
```bash
# Build Docker image
docker build -t chronos-deep:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name chronos-app chronos-deep:latest
```

### 3. Hugging Face Spaces (16 GB Free RAM)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Docker SDK** (Blank).
3. Set your Space remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/time-series-forecasting
   git push space main
   ```

---

## 📄 License & Contact

Distributed under the MIT License. See `LICENSE` for more information.

* **Author:** Akhil Vikram Singh
* **GitHub:** [@sAkhil2027](https://github.com/sAkhil2027)
* **Repository:** [Time_Series_Forecasting](https://github.com/sAkhil2027/Time_Series_Forecasting.git)
