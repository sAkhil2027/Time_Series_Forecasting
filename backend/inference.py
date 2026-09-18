import os
import json
import numpy as np
from datetime import timedelta
import pandas as pd
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

_models = {}
_data_cache = None
_metrics_cache = None

def get_metrics():
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            _metrics_cache = json.load(f)
            return _metrics_cache
    return {
        'MLP': {'val_rmse': 18.50, 'description': 'Multilayer Perceptron (Dense Layers)'},
        'CNN': {'val_rmse': 18.76, 'description': '1D Convolutional Neural Network'},
        'LSTM': {'val_rmse': 18.76, 'description': 'Long Short-Term Memory Network'},
        'CNN-LSTM': {'val_rmse': 19.17, 'description': 'Hybrid TimeDistributed CNN-LSTM'}
    }

def get_data():
    global _data_cache
    if _data_cache is None:
        print("[Inference] Loading sales dataset into memory...")
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        recent_df = df[df['date'] >= '2016-01-01'].copy()
        _data_cache = recent_df.sort_values('date')
    return _data_cache

def load_model_cached(model_name: str):
    global _models
    model_name = model_name.upper().replace('_', '-')
    if model_name in _models:
        return _models[model_name]

    filename_map = {
        'MLP': 'model_mlp.keras',
        'CNN': 'model_cnn.keras',
        'LSTM': 'model_lstm.keras',
        'CNN-LSTM': 'model_cnn_lstm.keras'
    }

    if model_name not in filename_map:
        raise ValueError(f"Unknown model name: {model_name}")

    model_file = os.path.join(SAVED_MODELS_DIR, filename_map[model_name])
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}.")

    print(f"[Inference] Loading model {model_name} from {model_file}...")
    model = tf.keras.models.load_model(model_file)
    _models[model_name] = model
    return model

def format_input_for_model(sequence_30: np.ndarray, model_name: str) -> np.ndarray:
    arr = np.array(sequence_30, dtype=np.float32).reshape(1, 30)
    model_name = model_name.upper().replace('_', '-')

    if model_name == 'MLP':
        return arr
    elif model_name in ['CNN', 'LSTM']:
        return arr.reshape((1, 30, 1))
    elif model_name == 'CNN-LSTM':
        return arr.reshape((1, 2, 15, 1))
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def predict_single_step(model, sequence_30: np.ndarray, model_name: str) -> float:
    inp = format_input_for_model(sequence_30, model_name)
    pred = model(inp, training=False).numpy()
    val = float(pred[0][0])
    return max(0.0, val)

def get_store_item_history(store_id: int, item_id: int, days: int = 90):
    df = get_data()
    sub = df[(df['store'] == store_id) & (df['item'] == item_id)].sort_values('date')
    if len(sub) == 0:
        return []
    
    recent = sub.tail(days)
    records = []
    for _, row in recent.iterrows():
        records.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'sales': float(row['sales'])
        })
    return records

def run_forecast(store_id: int, item_id: int, model_name: str, horizon: int = 30):
    df = get_data()
    sub = df[(df['store'] == store_id) & (df['item'] == item_id)].sort_values('date')
    if len(sub) < 30:
        raise ValueError(f"Insufficient historical data for Store {store_id}, Item {item_id}")

    last_30_records = sub.tail(30)
    window = list(last_30_records['sales'].values)
    last_date = last_30_records['date'].max()

    model = load_model_cached(model_name)
    forecast_points = []
    curr_window = list(window)

    for step in range(1, horizon + 1):
        step_date = last_date + timedelta(days=step)
        pred_val = predict_single_step(model, np.array(curr_window[-30:]), model_name)
        forecast_points.append({
            'date': step_date.strftime('%Y-%m-%d'),
            'sales': round(pred_val, 2)
        })
        curr_window.append(pred_val)

    return {
        'model': model_name,
        'store_id': store_id,
        'item_id': item_id,
        'horizon': horizon,
        'forecast': forecast_points,
        'summary': {
            'total_projected_sales': round(sum(p['sales'] for p in forecast_points), 1),
            'avg_daily_sales': round(np.mean([p['sales'] for p in forecast_points]), 2),
            'peak_day': max(forecast_points, key=lambda x: x['sales'])['date'] if forecast_points else None
        }
    }

def compare_all_models(store_id: int, item_id: int, horizon: int = 30):
    models = ['MLP', 'CNN', 'LSTM', 'CNN-LSTM']
    results = {}
    metrics = get_metrics()

    for m in models:
        try:
            fc = run_forecast(store_id, item_id, m, horizon)
            results[m] = {
                'forecast': fc['forecast'],
                'summary': fc['summary'],
                'val_rmse': metrics.get(m, {}).get('val_rmse', 'N/A'),
                'description': metrics.get(m, {}).get('description', '')
            }
        except Exception as e:
            results[m] = {
                'error': str(e),
                'val_rmse': metrics.get(m, {}).get('val_rmse', 'N/A'),
                'description': metrics.get(m, {}).get('description', '')
            }

    return results
