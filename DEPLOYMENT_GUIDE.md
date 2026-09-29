# 🚀 Deployment Guide: Deep Learning Time Series Web Application

This guide walks you through running your application locally and deploying it live on the web for free.

---

## 💻 Running Locally

To run the application on your computer:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Train and export models (if not already completed)
python backend/train_export_models.py

# 3. Launch the FastAPI web server
uvicorn backend.app:app --reload --port 8000
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🌐 Deploy Live to the Web (Free Cloud Hosting)

### Method 1: Render.com (Recommended & Easiest)

Render connects directly to your GitHub repository and automatically deploys whenever you push changes.

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add FastAPI backend and interactive frontend"
   git push origin main
   ```
2. **Create a Free Account on [Render.com](https://render.com/)**.
3. Click **"New +"** -> **"Web Service"**.
4. Connect your GitHub repository.
5. Fill in the settings:
   * **Name:** `time-series-forecasting`
   * **Language:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   * **Plan:** Free
6. Click **"Create Web Service"**.
   * Render will automatically install dependencies, build the container, and provide a public live URL (e.g., `https://time-series-forecasting.onrender.com`).

---

### Method 2: Hugging Face Spaces (100% Free - 16GB RAM)

Hugging Face Spaces is great for TensorFlow deep learning models because it offers 16GB RAM for free.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Space name: `time-series-forecasting`
3. Space SDK: Select **"Docker"** -> **"Blank"**.
4. Choose **Public**.
5. Push your repository code to the Hugging Face Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/time-series-forecasting
   git push space main
   ```
6. Hugging Face will automatically use the included [`Dockerfile`](file:///f:/Time%20series%20forecasting%20using%20Deep%20Learning/Time-series-forecasting-using-Deep-Learning-main/Time-series-forecasting-using-Deep-Learning-main/Dockerfile) and launch your web app live.

---

### Method 3: Railway.app

1. Sign up at [Railway.app](https://railway.app/).
2. Click **"New Project"** -> **"Deploy from GitHub repo"**.
3. Select this repository.
4. Railway detects Python and automatically deploys the application.

---

## 📁 Project Architecture Summary

```
├── backend/
│   ├── app.py                     # FastAPI server & REST API
│   ├── inference.py               # Preprocessing & autoregressive forecasting engine
│   └── train_export_models.py     # Deep learning model training & export
├── saved_models/                  # Saved .keras models & performance metrics
│   ├── model_mlp.keras
│   ├── model_cnn.keras
│   ├── model_lstm.keras
│   ├── model_cnn_lstm.keras
│   └── metrics.json
├── static/                        # Frontend UI
│   ├── index.html                 # Semantic HTML5 layout
│   ├── css/styles.css             # Glassmorphism dark mode styling
│   └── js/app.js                  # Dynamic Chart.js and API connection
├── Dockerfile                     # Cloud container configuration
├── requirements.txt               # Pinned Python dependencies
└── render.yaml                    # 1-click deploy blueprint for Render
```
