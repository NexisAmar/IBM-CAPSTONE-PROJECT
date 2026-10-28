"""
Unit tests for the model module.

Tests are isolated from production by:
  - Using temp directories for model storage
  - Training on small synthetic datasets
  - Not touching production model files
"""

import os
import sys
import shutil
import unittest
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.logger import set_log_dir, reset_log_dir
from src.model import (
    engineer_features, get_feature_columns, baseline_predict,
    train_model, compare_models, save_model, load_model, predict,
    set_model_dir, reset_model_dir,
)


class TestModel(unittest.TestCase):
    """Test suite for the model module."""

    @classmethod
    def setUpClass(cls):
        """Create test dataset and isolated directories."""
        cls._test_dir = tempfile.mkdtemp(prefix="aavail_test_model_")
        cls._test_model_dir = os.path.join(cls._test_dir, "models")
        cls._test_log_dir = os.path.join(cls._test_dir, "logs")
        os.makedirs(cls._test_model_dir, exist_ok=True)
        os.makedirs(cls._test_log_dir, exist_ok=True)

        set_model_dir(cls._test_model_dir)
        set_log_dir(cls._test_log_dir)

        # Create synthetic time series
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", "2019-06-30", freq="D")
        revenue = (
            1000
            + np.arange(len(dates)) * 2  # trend
            + 300 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # seasonality
            + np.random.normal(0, 100, len(dates))  # noise
        )
        cls.ts_df = pd.DataFrame({"date": dates, "revenue": np.maximum(revenue, 0)})

    @classmethod
    def tearDownClass(cls):
        reset_model_dir()
        reset_log_dir()
        shutil.rmtree(cls._test_dir, ignore_errors=True)

    def test_engineer_features_shape(self):
        """Test that feature engineering produces expected columns."""
        df = engineer_features(self.ts_df)
        expected_cols = get_feature_columns()
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing feature column: {col}")
        self.assertGreater(len(df), 0)

    def test_engineer_features_no_nulls(self):
        """Test that engineered features have no NaN values."""
        df = engineer_features(self.ts_df)
        feature_cols = get_feature_columns()
        for col in feature_cols:
            self.assertFalse(df[col].isna().any(), f"NaN found in column: {col}")

    def test_baseline_predict(self):
        """Test baseline (SMA) model produces predictions."""
        result = baseline_predict(self.ts_df, window=7)
        self.assertIn("prediction", result.columns)
        self.assertEqual(len(result), len(self.ts_df))
        self.assertFalse(result["prediction"].isna().all())

    def test_train_random_forest(self):
        """Test training a Random Forest model."""
        result = train_model(self.ts_df, model_type="random_forest", country="test")
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("rmse", result["metrics"])
        self.assertIn("mae", result["metrics"])
        self.assertGreater(result["metrics"]["rmse"], 0)
        self.assertIsNotNone(result["model"])

    def test_train_gradient_boosting(self):
        """Test training a Gradient Boosting model."""
        result = train_model(self.ts_df, model_type="gradient_boosting", country="test")
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"]["rmse"], 0)

    def test_compare_models(self):
        """Test that compare_models returns multiple results sorted by RMSE."""
        results = compare_models(self.ts_df, country="test")
        self.assertGreaterEqual(len(results), 2)
        # Verify sorted by RMSE
        rmse_vals = [r["metrics"]["rmse"] for r in results]
        self.assertEqual(rmse_vals, sorted(rmse_vals))

    def test_save_and_load_model(self):
        """Test model save/load roundtrip."""
        result = train_model(self.ts_df, model_type="random_forest", country="test_save")
        save_model(result, country="test_save")

        loaded = load_model(country="test_save", model_type="random_forest")
        self.assertIn("model", loaded)
        self.assertIsNotNone(loaded["model"])
        self.assertIn("features", loaded)

    def test_load_nonexistent_model(self):
        """Test that loading a nonexistent model raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_model(country="nonexistent_country_xyz")

    def test_predict_returns_nonnegative(self):
        """Test that predictions are non-negative."""
        result = train_model(self.ts_df, model_type="random_forest", country="test_pred")
        pred = predict(self.ts_df, result["model"], "2019-06-15")
        self.assertGreaterEqual(pred, 0)
        self.assertIsInstance(pred, float)

    def test_predict_future_date(self):
        """Test prediction for a date beyond the training data."""
        result = train_model(self.ts_df, model_type="random_forest", country="test_future")
        pred = predict(self.ts_df, result["model"], "2019-12-25")
        self.assertGreaterEqual(pred, 0)
        self.assertIsInstance(pred, float)

    def test_invalid_model_type(self):
        """Test that an invalid model type raises ValueError."""
        with self.assertRaises(ValueError):
            train_model(self.ts_df, model_type="invalid_model")

    def test_model_metadata(self):
        """Test that training returns proper metadata."""
        result = train_model(self.ts_df, model_type="random_forest", country="test_meta")
        meta = result["metadata"]
        self.assertIn("model_type", meta)
        self.assertIn("country", meta)
        self.assertIn("training_date", meta)
        self.assertIn("runtime_seconds", meta)
        self.assertEqual(meta["model_type"], "random_forest")
        self.assertEqual(meta["country"], "test_meta")


if __name__ == "__main__":
    unittest.main(verbosity=2)
