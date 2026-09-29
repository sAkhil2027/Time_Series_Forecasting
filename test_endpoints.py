import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        print("[Endpoint Test] /api/health passed:", data)

    def test_metadata(self):
        res = self.client.get("/api/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("stores", data)
        self.assertIn("items", data)
        self.assertIn("models", data)
        print("[Endpoint Test] /api/metadata passed with models:", [m["id"] for m in data["models"]])

    def test_history(self):
        res = self.client.get("/api/history?store_id=1&item_id=1&days=30")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["data"]), 30)
        print("[Endpoint Test] /api/history passed, points:", len(data["data"]))

    def test_forecast(self):
        payload = {"store_id": 1, "item_id": 1, "model_name": "LSTM", "horizon": 15}
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model"], "LSTM")
        self.assertEqual(len(data["forecast"]), 15)
        print("[Endpoint Test] /api/forecast passed. Total projected:", data["summary"]["total_projected_sales"])

    def test_compare(self):
        payload = {"store_id": 2, "item_id": 3, "horizon": 7}
        res = self.client.post("/api/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("comparisons", data)
        self.assertIn("CNN-LSTM", data["comparisons"])
        print("[Endpoint Test] /api/compare passed.")

    def test_custom_forecast(self):
        payload = {"values": [float(i * 1.5) for i in range(30)], "model_name": "CNN", "horizon": 10}
        res = self.client.post("/api/forecast-custom", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["forecast"]), 10)
        print("[Endpoint Test] /api/forecast-custom passed.")

    def test_upload_csv(self):
        csv_content = "date,sales\n" + "\n".join([f"2023-01-{i:02d},{i*2}" for i in range(1, 35)])
        files = {"file": ("test_sales.csv", csv_content.encode("utf-8"), "text/csv")}
        res = self.client.post("/api/upload-csv?model_name=LSTM&horizon=7", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["detected_column"], "sales")
        self.assertEqual(len(data["result"]["forecast"]), 7)
        print("[Endpoint Test] /api/upload-csv passed.")

    def test_static_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("ChronosDeep", res.text)
        print("[Endpoint Test] / (HTML index) passed.")

if __name__ == '__main__':
    unittest.main()
