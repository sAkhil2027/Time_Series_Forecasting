import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.inference import get_metrics, get_data, load_model_cached

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing Time Series Forecasting Engine...")
    try:
        get_metrics()
        get_data()
        load_model_cached("LSTM")
        print("[Startup] Caches and default neural model initialized successfully.")
    except Exception as e:
        print(f"[Startup Warning] Model/Data preloading notice: {e}")
    yield
    print("[Shutdown] Shutting down Time Series Forecasting Engine.")

app = FastAPI(
    title="Deep Learning Time Series Forecasting API",
    description="Interactive forecasting API comparing MLP, CNN, LSTM, and CNN-LSTM models on store item demand.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Deep Learning Time Series Forecasting",
        "version": "1.0.0"
    }

@app.get("/api/metadata")
def api_metadata():
    metrics = get_metrics()
    return {
        "stores": list(range(1, 11)),
        "items": list(range(1, 51)),
        "models": [
            {"id": "LSTM", "name": "LSTM Network", "type": "Recurrent", "val_rmse": metrics.get("LSTM", {}).get("val_rmse", 18.76)},
            {"id": "CNN-LSTM", "name": "CNN-LSTM Hybrid", "type": "Hybrid Spatiotemporal", "val_rmse": metrics.get("CNN-LSTM", {}).get("val_rmse", 19.17)},
            {"id": "CNN", "name": "1D CNN", "type": "Convolutional", "val_rmse": metrics.get("CNN", {}).get("val_rmse", 18.76)},
            {"id": "MLP", "name": "MLP Baseline", "type": "Dense Feedforward", "val_rmse": metrics.get("MLP", {}).get("val_rmse", 18.50)}
        ],
        "default_horizon": 30,
        "max_horizon": 90
    }
