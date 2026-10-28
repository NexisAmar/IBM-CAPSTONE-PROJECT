"""
Model module for the AAVAIL revenue prediction pipeline.

Implements multiple forecasting models:
  1. Baseline: Simple Moving Average (SMA)
  2. Random Forest Regressor with engineered time features
  3. Gradient Boosting Regressor with engineered time features

Provides functions to:
  - Engineer time-series features
  - Train and compare multiple models
  - Save/load models
  - Make predictions for specific dates and countries
  - Evaluate model performance (RMSE, MAE)
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import log_train, log_general

# Default directories
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Module-level override for test isolation
_model_dir_override = None


def set_model_dir(path):
    """Override model directory for test isolation."""
    global _model_dir_override
    _model_dir_override = path
    os.makedirs(path, exist_ok=True)


def reset_model_dir():
    """Reset to default production model directory."""
    global _model_dir_override
    _model_dir_override = None


def _get_model_dir():
    """Return current model directory."""
    return _model_dir_override if _model_dir_override else MODEL_DIR


def engineer_features(ts_df):
    """
    Create time-series features from a date/revenue DataFrame.

    Features:
      - year, month, day, day_of_week, day_of_year, week_of_year
      - is_weekend, is_month_start, is_month_end
      - quarter, sin/cos cyclical month encoding
      - rolling means (7-day, 30-day)
      - lag features (1, 7, 30 days)
      - days since start (trend)

    Parameters
    ----------
    ts_df : pd.DataFrame
        Must have 'date' and 'revenue' columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix with engineered columns.
    """
    df = ts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Trend feature
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    # Rolling features
    df["rolling_mean_7"] = df["revenue"].rolling(window=7, min_periods=1).mean()
    df["rolling_mean_30"] = df["revenue"].rolling(window=30, min_periods=1).mean()
    df["rolling_std_7"] = df["revenue"].rolling(window=7, min_periods=1).std().fillna(0)

    # Lag features
    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df["revenue"].shift(lag)

    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns():
    """Return list of feature column names used for modeling."""
    return [
        "year", "month", "day", "day_of_week", "day_of_year", "week_of_year",
        "is_weekend", "is_month_start", "is_month_end", "quarter",
        "month_sin", "month_cos", "days_since_start",
        "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
        "lag_1", "lag_7", "lag_30",
    ]


def baseline_predict(ts_df, window=30):
    """
    Baseline model: Simple Moving Average.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Time-series with 'date' and 'revenue' columns.
    window : int
        Window size for the moving average.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date', 'revenue' (actual), and 'prediction' columns.
    """
    df = ts_df.copy()
    df["prediction"] = df["revenue"].rolling(window=window, min_periods=1).mean()
    return df


def train_model(ts_df, model_type="random_forest", country="all"):
    """
    Train a model on the given time-series data.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Time-series data with 'date' and 'revenue' columns.
    model_type : str
        One of 'random_forest', 'gradient_boosting'.
    country : str
        Country label for logging and file naming.

    Returns
    -------
    dict
        {
            'model': fitted sklearn Pipeline,
            'features': list of feature names,
            'metrics': dict with 'rmse' and 'mae',
            'metadata': dict with training info,
        }
    """
    start_time = time.time()

    # Engineer features
    df = engineer_features(ts_df)
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df["revenue"].values

    # Time-series split for evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == "random_forest":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                ))
            ])
        elif model_type == "gradient_boosting":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    # Final model trained on all data
    if model_type == "random_forest":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ))
        ])
    else:
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ))
        ])

    final_model.fit(X, y)

    runtime = time.time() - start_time
    avg_rmse = float(np.mean(rmse_scores))
    avg_mae = float(np.mean(mae_scores))

    # Log training event
    log_train(country, model_type, "rmse", avg_rmse, runtime, ts_df.shape)
    log_general(f"Trained {model_type} for {country}: RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}")

    result = {
        "model": final_model,
        "features": feature_cols,
        "metrics": {"rmse": avg_rmse, "mae": avg_mae},
        "metadata": {
            "model_type": model_type,
            "country": country,
            "training_date": pd.Timestamp.now().isoformat(),
            "data_shape": list(ts_df.shape),
            "runtime_seconds": round(runtime, 4),
        }
    }
    return result


def compare_models(ts_df, country="all"):
    """
    Train and compare multiple models, returning results sorted by RMSE.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Time-series data.
    country : str
        Country label.

    Returns
    -------
    list[dict]
        List of model result dicts sorted by RMSE (best first).
    """
    results = []
    for model_type in ["random_forest", "gradient_boosting"]:
        print(f"  Training {model_type} for {country}...")
        result = train_model(ts_df, model_type=model_type, country=country)
        results.append(result)
        print(f"    RMSE: {result['metrics']['rmse']:.2f}, MAE: {result['metrics']['mae']:.2f}")

    # Sort by RMSE (best first)
    results.sort(key=lambda x: x["metrics"]["rmse"])
    return results


def save_model(result, country="all"):
    """
    Save a trained model and its metadata to disk.

    Parameters
    ----------
    result : dict
        Output from train_model().
    country : str
        Country for filename.
    """
    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    model_type = result["metadata"]["model_type"]
    safe_country = country.lower().replace(" ", "_")
    model_path = os.path.join(model_dir, f"{safe_country}-{model_type}.joblib")
    meta_path = os.path.join(model_dir, f"{safe_country}-{model_type}-metadata.json")

    joblib.dump(result["model"], model_path)

    meta = {
        **result["metadata"],
        "metrics": result["metrics"],
        "features": result["features"],
        "model_path": model_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved: {model_path}")
    return model_path


def load_model(country="all", model_type="random_forest"):
    """
    Load a trained model from disk.

    Parameters
    ----------
    country : str
        Country name.
    model_type : str
        Model type.

    Returns
    -------
    dict
        {'model': sklearn Pipeline, 'metadata': dict, 'features': list}
    """
    model_dir = _get_model_dir()
    safe_country = country.lower().replace(" ", "_")
    model_path = os.path.join(model_dir, f"{safe_country}-{model_type}.joblib")
    meta_path = os.path.join(model_dir, f"{safe_country}-{model_type}-metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return {
        "model": model,
        "metadata": metadata,
        "features": metadata.get("features", get_feature_columns()),
    }


def predict(ts_df, model, target_date):
    """
    Make a revenue prediction for a specific date.

    Needs the historical time-series to compute rolling/lag features.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Historical time-series with 'date' and 'revenue'.
    model : sklearn Pipeline
        Trained model.
    target_date : str or pd.Timestamp
        Date to predict.

    Returns
    -------
    float
        Predicted revenue.
    """
    target_date = pd.Timestamp(target_date)

    # Extend time series to target date if needed
    last_date = ts_df["date"].max()
    if target_date > last_date:
        extra_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=target_date, freq="D")
        extra_df = pd.DataFrame({"date": extra_dates, "revenue": 0.0})
        ts_df = pd.concat([ts_df, extra_df], ignore_index=True)

    df = engineer_features(ts_df)
    feature_cols = get_feature_columns()

    # Find the row for target_date
    mask = df["date"] == target_date
    if mask.sum() == 0:
        # Use the last available row as proxy
        row = df.iloc[-1:]
    else:
        row = df[mask]

    X = row[feature_cols].values
    prediction = float(model.predict(X)[0])
    return max(0.0, prediction)  # Revenue can't be negative


if __name__ == "__main__":
    from src.ingest import ingest_data

    print("Loading data...")
    ts, countries = ingest_data()

    print("\nComparing models for 'all' countries...")
    results = compare_models(ts, country="all")

    best = results[0]
    print(f"\nBest model: {best['metadata']['model_type']}")
    print(f"  RMSE: {best['metrics']['rmse']:.2f}")
    print(f"  MAE:  {best['metrics']['mae']:.2f}")

    save_model(best, country="all")
