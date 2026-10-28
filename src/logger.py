"""
Logging module for the AAVAIL revenue prediction pipeline.

Provides structured logging for:
  - API requests (predict & train)
  - Model training events
  - Performance monitoring events
  - General application events

Logs are stored as JSON-lines files for easy parsing and analysis.
Production and test logs are fully isolated via configurable log directory.
"""

import os
import json
import time
import uuid
from datetime import datetime

# Default log directory (production)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Module-level state for log directory override (used by tests)
_log_dir_override = None


def set_log_dir(path):
    """Override the log directory (used for test isolation)."""
    global _log_dir_override
    _log_dir_override = path
    os.makedirs(path, exist_ok=True)


def reset_log_dir():
    """Reset to default production log directory."""
    global _log_dir_override
    _log_dir_override = None


def _get_log_dir():
    """Get the current active log directory."""
    return _log_dir_override if _log_dir_override else LOG_DIR


def _ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = _get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _write_log(log_type, data):
    """
    Write a log entry to the appropriate log file.

    Parameters
    ----------
    log_type : str
        One of 'train', 'predict', 'monitor', 'general'.
    data : dict
        Data to log.
    """
    log_dir = _ensure_log_dir()
    log_file = os.path.join(log_dir, f"{log_type}-log.json")

    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "type": log_type,
        **data,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def log_train(country, model_name, metric_name, metric_value, runtime_seconds, data_shape):
    """Log a model training event."""
    return _write_log("train", {
        "country": country,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": round(metric_value, 4) if isinstance(metric_value, float) else metric_value,
        "runtime_seconds": round(runtime_seconds, 4),
        "data_shape": list(data_shape) if hasattr(data_shape, '__iter__') else data_shape,
    })


def log_predict(country, date, revenue_prediction, model_name, runtime_seconds):
    """Log a prediction request."""
    return _write_log("predict", {
        "country": country,
        "date": str(date),
        "revenue_prediction": round(revenue_prediction, 2) if isinstance(revenue_prediction, float) else revenue_prediction,
        "model_name": model_name,
        "runtime_seconds": round(runtime_seconds, 6),
    })


def log_monitor(country, metric_name, metric_value, threshold=None, alert=False):
    """Log a performance monitoring event."""
    return _write_log("monitor", {
        "country": country,
        "metric_name": metric_name,
        "metric_value": round(metric_value, 6) if isinstance(metric_value, float) else metric_value,
        "threshold": threshold,
        "alert": alert,
    })


def log_general(message, level="INFO"):
    """Log a general application event."""
    return _write_log("general", {
        "message": message,
        "level": level,
    })


def read_logs(log_type, n=None):
    """
    Read log entries from a log file.

    Parameters
    ----------
    log_type : str
        One of 'train', 'predict', 'monitor', 'general'.
    n : int or None
        Number of most recent entries to return (None for all).

    Returns
    -------
    list[dict]
        List of log entries.
    """
    log_dir = _get_log_dir()
    log_file = os.path.join(log_dir, f"{log_type}-log.json")

    if not os.path.exists(log_file):
        return []

    entries = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if n is not None:
        entries = entries[-n:]

    return entries


if __name__ == "__main__":
    # Example usage
    log_general("Application started")
    log_train("United Kingdom", "RandomForest", "rmse", 1234.56, 2.5, (500, 5))
    log_predict("United Kingdom", "2019-08-01", 45678.90, "RandomForest", 0.03)
    log_monitor("United Kingdom", "wasserstein_distance", 0.05, threshold=0.1, alert=False)

    print("Recent train logs:")
    for entry in read_logs("train", n=5):
        print(f"  {entry}")
