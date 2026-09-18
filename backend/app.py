import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
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

from backend.inference import get_store_item_history

@app.get("/api/history")
def api_history(store_id: int = Query(1, ge=1, le=10), item_id: int = Query(1, ge=1, le=50), days: int = Query(90, ge=30, le=365)):
    try:
        history = get_store_item_history(store_id, item_id, days)
        return {"store_id": store_id, "item_id": item_id, "days": days, "data": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
from backend.inference import run_forecast, compare_all_models

class ForecastRequest(BaseModel):
    store_id: int = 1
    item_id: int = 1
    model_name: str = "LSTM"
    horizon: int = 30

class CompareRequest(BaseModel):
    store_id: int = 1
    item_id: int = 1
    horizon: int = 30

@app.post("/api/forecast")
def api_forecast(req: ForecastRequest):
    try:
        result = run_forecast(req.store_id, req.item_id, req.model_name, req.horizon)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=f"Model not ready: {str(fnf)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare")
def api_compare(req: CompareRequest):
    try:
        results = compare_all_models(req.store_id, req.item_id, req.horizon)
        return {
            "store_id": req.store_id,
            "item_id": req.item_id,
            "horizon": req.horizon,
            "comparisons": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
