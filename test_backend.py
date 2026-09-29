import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.inference import (
    get_metrics,
    get_data,
    load_model_cached,
    format_input_for_model,
    predict_single_step,
    get_store_item_history,
    run_forecast,
    compare_all_models,
    forecast_custom_sequence
)

class TestDeepLearningForecast(unittest.TestCase):

    def test_01_metrics(self):
        metrics = get_metrics()
        print("\n[Test] Loaded metrics:", list(metrics.keys()))
        self.assertIn('LSTM', metrics)
        self.assertIn('CNN', metrics)
        self.assertIn('MLP', metrics)
        self.assertIn('CNN-LSTM', metrics)

    def test_02_data_loading(self):
        df = get_data()
        print(f"[Test] Data loaded with {len(df)} rows. Columns: {df.columns.tolist()}")
        self.assertFalse(df.empty)
        self.assertIn('store', df.columns)
        self.assertIn('item', df.columns)
        self.assertIn('sales', df.columns)
        self.assertIn('date', df.columns)

    def test_03_model_loading_and_prediction(self):
        models = ['MLP', 'CNN', 'LSTM', 'CNN-LSTM']
        test_seq = np.random.uniform(20, 60, 30)
        for m in models:
            model = load_model_cached(m)
            self.assertIsNotNone(model)
            val = predict_single_step(model, test_seq, m)
            print(f"[Test] Model {m} output for random sequence: {val:.2f}")
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0.0)

    def test_04_store_item_history(self):
        history = get_store_item_history(store_id=1, item_id=1, days=30)
        self.assertEqual(len(history), 30)
        self.assertIn('date', history[0])
        self.assertIn('sales', history[0])
        print(f"[Test] History sample: {history[0]}")

    def test_05_run_forecast(self):
        res = run_forecast(store_id=1, item_id=1, model_name="LSTM", horizon=14)
        self.assertEqual(res['model'], "LSTM")
        self.assertEqual(len(res['forecast']), 14)
        self.assertIn('total_projected_sales', res['summary'])
        self.assertIn('avg_daily_sales', res['summary'])
        print(f"[Test] Forecast summary (14 days): {res['summary']}")

    def test_06_compare_all_models(self):
        res = compare_all_models(store_id=1, item_id=1, horizon=7)
        self.assertEqual(len(res), 4)
        for m in ['MLP', 'CNN', 'LSTM', 'CNN-LSTM']:
            self.assertIn(m, res)
            self.assertIn('forecast', res[m])
            self.assertEqual(len(res[m]['forecast']), 7)
        print("[Test] Compare all 4 models completed successfully.")

    def test_07_forecast_custom_sequence(self):
        custom_vals = [float(x) for x in range(1, 35)]
        res = forecast_custom_sequence(custom_vals, model_name="CNN", horizon=5)
        self.assertEqual(len(res['forecast']), 5)
        self.assertEqual(res['model'], "CNN")
        print(f"[Test] Custom sequence forecast: {res['forecast']}")

if __name__ == '__main__':
    unittest.main()
