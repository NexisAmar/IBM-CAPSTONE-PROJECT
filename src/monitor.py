"""
Performance monitoring module for the AAVAIL revenue prediction pipeline.

Implements:
  - Wasserstein distance for distribution drift detection
  - Prediction error tracking over time
  - Model staleness detection
  - Summary statistics reporting
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from datetime import datetime, timedelta

from src.logger import log_monitor, read_logs


# Thresholds for alerts
WASSERSTEIN_THRESHOLD = 0.15
RMSE_DRIFT_THRESHOLD = 1.5  # multiplier of baseline RMSE
STALENESS_DAYS = 30


def compute_wasserstein_drift(reference_data, current_data, country="all"):
    """
    Compute the Wasserstein distance between reference and current revenue distributions.

    Parameters
    ----------
    reference_data : array-like
        Historical revenue values (training distribution).
    current_data : array-like
        Recent revenue values (inference distribution).
    country : str
        Country label for logging.

    Returns
    -------
    dict
        {'distance': float, 'alert': bool, 'threshold': float}
    """
    ref = np.array(reference_data, dtype=float)
    cur = np.array(current_data, dtype=float)

    # Normalize to [0, 1] for comparable scale
    combined_max = max(ref.max(), cur.max())
    if combined_max > 0:
        ref_norm = ref / combined_max
        cur_norm = cur / combined_max
    else:
        ref_norm = ref
        cur_norm = cur

    distance = float(wasserstein_distance(ref_norm, cur_norm))
    alert = distance > WASSERSTEIN_THRESHOLD

    log_monitor(country, "wasserstein_distance", distance,
                threshold=WASSERSTEIN_THRESHOLD, alert=alert)

    return {
        "distance": round(distance, 6),
        "alert": alert,
        "threshold": WASSERSTEIN_THRESHOLD,
    }


def compute_prediction_error(actual, predicted, country="all"):
    """
    Compute prediction errors and log them.

    Parameters
    ----------
    actual : array-like
        Actual revenue values.
    predicted : array-like
        Predicted revenue values.
    country : str
        Country label.

    Returns
    -------
    dict
        {'rmse': float, 'mae': float, 'mape': float}
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    # MAPE (avoid division by zero)
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100)
    else:
        mape = 0.0

    log_monitor(country, "prediction_rmse", rmse)
    log_monitor(country, "prediction_mae", mae)
    log_monitor(country, "prediction_mape", mape)

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}


def check_model_staleness(metadata, country="all"):
    """
    Check if a model is stale based on training date.

    Parameters
    ----------
    metadata : dict
        Model metadata with 'training_date' key.
    country : str
        Country label.

    Returns
    -------
    dict
        {'is_stale': bool, 'days_since_training': int, 'threshold_days': int}
    """
    training_date = pd.Timestamp(metadata.get("training_date", "2020-01-01"))
    days_since = (pd.Timestamp.now() - training_date).days
    is_stale = days_since > STALENESS_DAYS

    log_monitor(country, "model_staleness_days", days_since,
                threshold=STALENESS_DAYS, alert=is_stale)

    return {
        "is_stale": is_stale,
        "days_since_training": days_since,
        "threshold_days": STALENESS_DAYS,
    }


def get_monitoring_summary():
    """
    Generate a summary of recent monitoring events.

    Returns
    -------
    dict
        Summary of recent monitoring metrics.
    """
    monitor_logs = read_logs("monitor", n=50)

    if not monitor_logs:
        return {"status": "no_data", "message": "No monitoring data available"}

    # Group by metric name
    metrics = {}
    for entry in monitor_logs:
        name = entry.get("metric_name", "unknown")
        value = entry.get("metric_value", 0)
        if name not in metrics:
            metrics[name] = []
        metrics[name].append(value)

    summary = {
        "status": "ok",
        "total_events": len(monitor_logs),
        "alerts": sum(1 for e in monitor_logs if e.get("alert", False)),
        "metrics": {}
    }

    for name, values in metrics.items():
        summary["metrics"][name] = {
            "count": len(values),
            "mean": round(float(np.mean(values)), 6),
            "min": round(float(np.min(values)), 6),
            "max": round(float(np.max(values)), 6),
        }

    return summary


if __name__ == "__main__":
    # Demo monitoring
    ref = np.random.normal(1000, 200, 100)
    cur = np.random.normal(1050, 220, 100)

    result = compute_wasserstein_drift(ref, cur, country="United Kingdom")
    print(f"Wasserstein drift: {result}")

    errors = compute_prediction_error(ref[:50], cur[:50], country="United Kingdom")
    print(f"Prediction errors: {errors}")

    summary = get_monitoring_summary()
    print(f"Monitoring summary: {summary}")
