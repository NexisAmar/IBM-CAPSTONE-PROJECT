"""
Unit tests for the logging module.

Tests are fully isolated from production by:
  - Using a temporary directory for all log files
  - Resetting the log directory after tests
  - Verifying no production log files are modified
"""

import os
import sys
import json
import shutil
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.logger import (
    log_train, log_predict, log_monitor, log_general,
    read_logs, set_log_dir, reset_log_dir, _get_log_dir,
)


class TestLogging(unittest.TestCase):
    """Test suite for the logging module."""

    @classmethod
    def setUpClass(cls):
        """Create isolated temp directory for test logs."""
        cls._test_dir = tempfile.mkdtemp(prefix="aavail_test_logs_")
        set_log_dir(cls._test_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up and reset."""
        reset_log_dir()
        shutil.rmtree(cls._test_dir, ignore_errors=True)

    def test_log_dir_isolation(self):
        """Verify that log directory is pointing to the test directory."""
        current_dir = _get_log_dir()
        self.assertEqual(current_dir, self._test_dir)
        self.assertNotIn("production", current_dir.lower())

    def test_log_train(self):
        """Test logging a training event."""
        entry = log_train(
            country="TestCountry",
            model_name="TestModel",
            metric_name="rmse",
            metric_value=123.456,
            runtime_seconds=1.5,
            data_shape=(100, 5),
        )
        self.assertIn("id", entry)
        self.assertIn("timestamp", entry)
        self.assertEqual(entry["type"], "train")
        self.assertEqual(entry["country"], "TestCountry")
        self.assertEqual(entry["model_name"], "TestModel")
        self.assertAlmostEqual(entry["metric_value"], 123.456, places=3)

    def test_log_predict(self):
        """Test logging a prediction event."""
        entry = log_predict(
            country="TestCountry",
            date="2019-08-01",
            revenue_prediction=5000.50,
            model_name="RandomForest",
            runtime_seconds=0.05,
        )
        self.assertEqual(entry["type"], "predict")
        self.assertEqual(entry["country"], "TestCountry")
        self.assertEqual(entry["revenue_prediction"], 5000.50)

    def test_log_monitor(self):
        """Test logging a monitoring event."""
        entry = log_monitor(
            country="TestCountry",
            metric_name="wasserstein_distance",
            metric_value=0.08,
            threshold=0.15,
            alert=False,
        )
        self.assertEqual(entry["type"], "monitor")
        self.assertFalse(entry["alert"])

    def test_log_monitor_with_alert(self):
        """Test logging a monitoring event with alert=True."""
        entry = log_monitor(
            country="TestCountry",
            metric_name="wasserstein_distance",
            metric_value=0.25,
            threshold=0.15,
            alert=True,
        )
        self.assertTrue(entry["alert"])

    def test_log_general(self):
        """Test logging a general event."""
        entry = log_general("Test message", level="INFO")
        self.assertEqual(entry["type"], "general")
        self.assertEqual(entry["message"], "Test message")
        self.assertEqual(entry["level"], "INFO")

    def test_log_general_error(self):
        """Test logging an error-level general event."""
        entry = log_general("Something broke", level="ERROR")
        self.assertEqual(entry["level"], "ERROR")

    def test_read_logs_train(self):
        """Test reading back train log entries."""
        # Write a fresh entry
        log_train("ReadTest", "RF", "rmse", 100.0, 0.5, (50, 3))
        entries = read_logs("train")
        self.assertGreater(len(entries), 0)
        last = entries[-1]
        self.assertEqual(last["country"], "ReadTest")

    def test_read_logs_with_limit(self):
        """Test reading logs with n limit."""
        for i in range(5):
            log_general(f"Limit test {i}")
        entries = read_logs("general", n=3)
        self.assertLessEqual(len(entries), 3)

    def test_read_logs_empty(self):
        """Test reading from a non-existent log type returns empty list."""
        entries = read_logs("nonexistent_type")
        self.assertEqual(entries, [])

    def test_log_files_created_in_test_dir(self):
        """Verify actual log files are created in the test directory."""
        log_general("File existence test")
        log_file = os.path.join(self._test_dir, "general-log.json")
        self.assertTrue(os.path.exists(log_file))

    def test_log_file_format_is_json_lines(self):
        """Verify log file format is valid JSON lines."""
        log_general("JSON format test")
        log_file = os.path.join(self._test_dir, "general-log.json")
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parsed = json.loads(line)  # Should not raise
                    self.assertIsInstance(parsed, dict)

    def test_log_entries_have_uuid(self):
        """Test that each log entry has a unique ID."""
        e1 = log_general("UUID test 1")
        e2 = log_general("UUID test 2")
        self.assertNotEqual(e1["id"], e2["id"])

    def test_log_entries_have_timestamp(self):
        """Test that each log entry has a timestamp."""
        entry = log_general("Timestamp test")
        self.assertIn("timestamp", entry)
        self.assertGreater(len(entry["timestamp"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
