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
