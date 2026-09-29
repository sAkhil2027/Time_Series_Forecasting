import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data():
    print("[1/5] Loading and filtering training data...")
    train = pd.read_csv(DATA_PATH, parse_dates=['date'])
    # Subsample to 2017 data as done in original notebook
    train_2017 = train[train['date'] >= '2017-01-01'].copy()
    
    train_gp = train_2017.sort_values('date').groupby(['item', 'store', 'date'], as_index=False).agg({'sales': 'mean'})
    
    window = 29
    lag = 90
    print(f"[2/5] Creating sliding windows (window={window}, lag={lag})...")
    series = series_to_supervised(train_gp.drop('date', axis=1), window=window, lag=lag)

    last_item = 'item(t-%d)' % window
    last_store = 'store(t-%d)' % window
    series = series[(series['store(t)'] == series[last_store])]
    series = series[(series['item(t)'] == series[last_item])]

    columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item', 'store']]
    for i in range(window, 0, -1):
        columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item', 'store']]
    series.drop(columns_to_drop, axis=1, inplace=True)
    series.drop(['item(t)', 'store(t)'], axis=1, inplace=True)

    labels_col = 'sales(t+%d)' % lag
    labels = series[labels_col]
    series = series.drop(labels_col, axis=1)

    X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
    print(f"Dataset prepared: Train {X_train.shape}, Validation {X_valid.shape}")
    return X_train, X_valid, Y_train, Y_valid

def train_and_save_all():
    X_train, X_valid, Y_train, Y_valid = prepare_data()

    X_train_val = X_train.values
    X_valid_val = X_valid.values

    X_train_series = X_train_val.reshape((X_train_val.shape[0], X_train_val.shape[1], 1))
    X_valid_series = X_valid_val.reshape((X_valid_val.shape[0], X_valid_val.shape[1], 1))

    subsequences = 2
    timesteps = X_train_series.shape[1] // subsequences
    X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
    X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))

    metrics = {}
    epochs = 12
    batch_size = 512
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 1. MLP Model
    print("\n--- Training MLP Model ---")
    model_mlp = Sequential([
        Dense(100, activation='relu', input_dim=X_train.shape[1]),
        Dense(1)
    ])
    model_mlp.compile(loss='mse', optimizer='adam')
    model_mlp.fit(X_train_val, Y_train, validation_data=(X_valid_val, Y_valid),
                  epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    val_pred = model_mlp.predict(X_valid_val, verbose=0)
    rmse = float(np.sqrt(mean_squared_error(Y_valid, val_pred)))
    metrics['MLP'] = {'val_rmse': round(rmse, 2), 'description': 'Multilayer Perceptron (Dense Feedforward)'}
    model_mlp.save(os.path.join(SAVED_MODELS_DIR, 'model_mlp.keras'))
    print(f"MLP Validation RMSE: {rmse:.2f} (Saved to saved_models/model_mlp.keras)")

    # 2. CNN Model
    print("\n--- Training CNN Model ---")
    model_cnn = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_series.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model_cnn.compile(loss='mse', optimizer='adam')
    model_cnn.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid),
                  epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    val_pred = model_cnn.predict(X_valid_series, verbose=0)
    rmse = float(np.sqrt(mean_squared_error(Y_valid, val_pred)))
    metrics['CNN'] = {'val_rmse': round(rmse, 2), 'description': '1D Convolutional Neural Network'}
    model_cnn.save(os.path.join(SAVED_MODELS_DIR, 'model_cnn.keras'))
    print(f"CNN Validation RMSE: {rmse:.2f} (Saved to saved_models/model_cnn.keras)")

    # 3. LSTM Model
    print("\n--- Training LSTM Model ---")
    model_lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], 1)),
        Dense(1)
    ])
    model_lstm.compile(loss='mse', optimizer='adam')
    model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid),
                  epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    val_pred = model_lstm.predict(X_valid_series, verbose=0)
    rmse = float(np.sqrt(mean_squared_error(Y_valid, val_pred)))
    metrics['LSTM'] = {'val_rmse': round(rmse, 2), 'description': 'Long Short-Term Memory Network'}
    model_lstm.save(os.path.join(SAVED_MODELS_DIR, 'model_lstm.keras'))
    print(f"LSTM Validation RMSE: {rmse:.2f} (Saved to saved_models/model_lstm.keras)")

    # 4. CNN-LSTM Hybrid Model
    print("\n--- Training CNN-LSTM Hybrid Model ---")
    model_cnn_lstm = Sequential([
        TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(subsequences, timesteps, 1)),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model_cnn_lstm.compile(loss='mse', optimizer='adam')
    model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid),
                       epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    val_pred = model_cnn_lstm.predict(X_valid_series_sub, verbose=0)
    rmse = float(np.sqrt(mean_squared_error(Y_valid, val_pred)))
    metrics['CNN-LSTM'] = {'val_rmse': round(rmse, 2), 'description': 'Hybrid TimeDistributed CNN-LSTM'}
    model_cnn_lstm.save(os.path.join(SAVED_MODELS_DIR, 'model_cnn_lstm.keras'))
    print(f"CNN-LSTM Validation RMSE: {rmse:.2f} (Saved to saved_models/model_cnn_lstm.keras)")

    # Save metrics metadata
    with open(os.path.join(SAVED_MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n[5/5] All models and metrics saved successfully to saved_models/ directory!")

if __name__ == '__main__':
    train_and_save_all()
