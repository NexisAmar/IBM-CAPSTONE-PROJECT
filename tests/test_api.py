"""
Unit tests for the Flask API endpoints.

Tests are isolated from production by:
  - Using Flask test client (no real HTTP server)
  - Using test-specific model and log directories
  - Cleaning up after each test
"""

import os
import sys
import json
import shutil
import unittest
import tempfile

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.logger import set_log_dir, reset_log_dir
from src.model import set_model_dir, reset_model_dir


class TestAPI(unittest.TestCase):
    """Test suite for the Flask API."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures: isolated dirs, generate data, train a model."""
        # Create isolated temp directories
        cls._test_dir = tempfile.mkdtemp(prefix="aavail_test_api_")
        cls._test_log_dir = os.path.join(cls._test_dir, "logs")
        cls._test_model_dir = os.path.join(cls._test_dir, "models")
        os.makedirs(cls._test_log_dir, exist_ok=True)
        os.makedirs(cls._test_model_dir, exist_ok=True)

        # Redirect logs and models to test directories
        set_log_dir(cls._test_log_dir)
        set_model_dir(cls._test_model_dir)

        # Generate test dataset
        from src.generate_data import generate_country_data
        import pandas as pd

        cls._test_data_dir = os.path.join(cls._test_dir, "data")
        os.makedirs(cls._test_data_dir, exist_ok=True)

        # Small dataset for fast tests
        records = generate_country_data("United Kingdom",
                                        start_date="2019-01-01",
                                        end_date="2019-03-31",
                                        seed=99)
        records += generate_country_data("Germany",
                                         start_date="2019-01-01",
                                         end_date="2019-03-31",
                                         seed=100)
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(cls._test_data_dir, "ts-data.csv"), index=False)

        # Train a test model
        from src.ingest import load_raw_data, clean_data, aggregate_revenue
        from src.model import train_model, save_model

        raw = load_raw_data(cls._test_data_dir)
        clean = clean_data(raw)
        ts = aggregate_revenue(clean, freq="D")
        result = train_model(ts, model_type="random_forest", country="all")
        save_model(result, country="all")

        # Also train for United Kingdom
        ts_uk = aggregate_revenue(clean, freq="D", country="United Kingdom")
        result_uk = train_model(ts_uk, model_type="random_forest", country="United Kingdom")
        save_model(result_uk, country="United Kingdom")

        # Import and configure the app
        import app as app_module
        app_module.app.config["TESTING"] = True

        # Patch the app's data loading to use test data
        from src.ingest import load_raw_data as _lr, clean_data as _cd, get_available_countries as _gc
        test_df = _lr(cls._test_data_dir)
        test_df = _cd(test_df)
        app_module._cache["data"] = test_df
        app_module._cache["countries"] = _gc(test_df)
        app_module._cache["models"] = {}

        cls.app = app_module.app
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        """Clean up test directories."""
        reset_log_dir()
        reset_model_dir()
        shutil.rmtree(cls._test_dir, ignore_errors=True)

    def test_health_check(self):
        """Test GET / returns health info."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("endpoints", data)

    def test_countries_endpoint(self):
        """Test GET /countries returns country list."""
        resp = self.client.get("/countries")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("countries", data)
        self.assertIn("all", data["countries"])
        self.assertIn("United Kingdom", data["countries"])

    def test_predict_all_countries(self):
        """Test POST /predict for all countries combined."""
        resp = self.client.post("/predict",
                                json={"country": "all", "date": "2019-03-15"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_revenue", data)
        self.assertIsInstance(data["predicted_revenue"], (int, float))
        self.assertGreaterEqual(data["predicted_revenue"], 0)
        self.assertEqual(data["country"], "all")

    def test_predict_specific_country(self):
        """Test POST /predict for a specific country (United Kingdom)."""
        resp = self.client.post("/predict",
                                json={"country": "United Kingdom", "date": "2019-03-15"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_revenue", data)
        self.assertGreaterEqual(data["predicted_revenue"], 0)
        self.assertEqual(data["country"], "United Kingdom")

    def test_predict_invalid_country(self):
        """Test prediction with an invalid country returns 400."""
        resp = self.client.post("/predict",
                                json={"country": "Atlantis", "date": "2019-03-15"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_predict_via_query_params(self):
        """Test GET /predict with query parameters."""
        resp = self.client.get("/predict?country=all&date=2019-03-15")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_revenue", data)

    def test_train_endpoint(self):
        """Test POST /train trains and returns metrics."""
        resp = self.client.post("/train",
                                json={"country": "all", "model_type": "random_forest"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("rmse", data)
        self.assertIn("mae", data)

    def test_train_compare(self):
        """Test POST /train with model_type=compare."""
        resp = self.client.post("/train",
                                json={"country": "all", "model_type": "compare"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("best_model", data)
        self.assertGreater(data["models_compared"], 1)

    def test_logs_endpoint(self):
        """Test GET /logs/<type> returns log entries."""
        for log_type in ["train", "predict", "general"]:
            resp = self.client.get(f"/logs/{log_type}?n=5")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("entries", data)

    def test_logs_invalid_type(self):
        """Test GET /logs/<invalid_type> returns 400."""
        resp = self.client.get("/logs/invalid")
        self.assertEqual(resp.status_code, 400)

    def test_monitor_endpoint(self):
        """Test GET /monitor returns monitoring data."""
        resp = self.client.get("/monitor")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
