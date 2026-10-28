"""
Flask API for the AAVAIL Revenue Prediction Service.

Endpoints:
  GET  /                 → Health check and API info
  POST /predict          → Get revenue predictions
  POST /train            → Train models
  GET  /logs/<log_type>  → Retrieve logs
  GET  /monitor          → Get monitoring summary
  GET  /countries        → List available countries
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.ingest import ingest_data, load_raw_data, clean_data, aggregate_revenue, get_available_countries
from src.model import (
    train_model, compare_models, save_model, load_model,
    predict, engineer_features, get_feature_columns, baseline_predict
)
from src.logger import log_predict, log_general, read_logs
from src.monitor import (
    compute_wasserstein_drift, compute_prediction_error,
    check_model_staleness, get_monitoring_summary
)

app = Flask(__name__)

# Cache for loaded data and models
_cache = {
    "data": None,
    "models": {},
    "countries": [],
}


def _ensure_data():
    """Load and cache the dataset."""
    if _cache["data"] is None:
        print("Loading dataset...")
        df = load_raw_data()
        df = clean_data(df)
        _cache["data"] = df
        _cache["countries"] = get_available_countries(df)
        log_general("Dataset loaded and cached")
    return _cache["data"]


def _get_model(country="all"):
    """Load or return cached model for a country."""
    if country not in _cache["models"]:
        try:
            result = load_model(country=country, model_type="random_forest")
            _cache["models"][country] = result
            log_general(f"Loaded model for {country}")
        except FileNotFoundError:
            return None
    return _cache["models"].get(country)


@app.route("/", methods=["GET"])
def index():
    """Health check and API info."""
    return jsonify({
        "service": "AAVAIL Revenue Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "POST /predict": "Get revenue predictions (params: country, date)",
            "POST /train": "Train models (params: country, model_type)",
            "GET /logs/<type>": "Retrieve logs (types: train, predict, monitor, general)",
            "GET /monitor": "Performance monitoring summary",
            "GET /countries": "List available countries",
        }
    })


@app.route("/predict", methods=["GET", "POST"])
def predict_endpoint():
    """
    Get revenue predictions.

    Query Parameters / JSON Body:
      - country (str): Country name or 'all' for combined. Default: 'all'.
      - date (str): Target date in YYYY-MM-DD format. Default: next day.

    Returns:
      JSON with prediction results.
    """
    start_time = time.time()

    try:
        # Parse parameters from query string or JSON body
        if request.method == "POST" and request.is_json:
            params = request.get_json()
        else:
            params = request.args.to_dict()

        country = params.get("country", "all")
        target_date = params.get("date", None)

        if target_date is None:
            target_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Load data
        df = _ensure_data()

        # Validate country
        valid_countries = ["all"] + _cache["countries"]
        if country not in valid_countries:
            return jsonify({
                "error": f"Unknown country: {country}",
                "valid_countries": valid_countries,
            }), 400

        # Get time series for the country
        ts = aggregate_revenue(df, freq="D", country=country if country != "all" else None)

        # Load model
        model_data = _get_model(country)
        if model_data is None:
            return jsonify({
                "error": f"No trained model found for '{country}'. Please train first via POST /train.",
                "suggestion": "POST /train with country parameter",
            }), 404

        # Make prediction
        prediction = predict(ts, model_data["model"], target_date)

        runtime = time.time() - start_time
        log_predict(country, target_date, prediction, "random_forest", runtime)

        return jsonify({
            "country": country,
            "date": target_date,
            "predicted_revenue": round(prediction, 2),
            "model": model_data.get("metadata", {}).get("model_type", "random_forest"),
            "model_metrics": model_data.get("metadata", {}).get("metrics", {}),
            "runtime_seconds": round(runtime, 4),
        })

    except Exception as e:
        log_general(f"Prediction error: {str(e)}", level="ERROR")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Train models for a specific country or all countries.

    Query Parameters / JSON Body:
      - country (str): Country name or 'all'. Default: 'all'.
      - model_type (str): 'random_forest', 'gradient_boosting', or 'compare'. Default: 'compare'.

    Returns:
      JSON with training results and metrics.
    """
    try:
        if request.is_json:
            params = request.get_json()
        else:
            params = request.args.to_dict()

        country = params.get("country", "all")
        model_type = params.get("model_type", "compare")

        # Load data
        df = _ensure_data()
        ts = aggregate_revenue(df, freq="D", country=country if country != "all" else None)

        if model_type == "compare":
            results = compare_models(ts, country=country)
            best = results[0]
            save_model(best, country=country)

            # Invalidate cache
            _cache["models"].pop(country, None)

            response = {
                "status": "success",
                "country": country,
                "models_compared": len(results),
                "best_model": best["metadata"]["model_type"],
                "results": [
                    {
                        "model_type": r["metadata"]["model_type"],
                        "rmse": round(r["metrics"]["rmse"], 2),
                        "mae": round(r["metrics"]["mae"], 2),
                        "runtime": round(r["metadata"]["runtime_seconds"], 2),
                    }
                    for r in results
                ],
            }
        else:
            result = train_model(ts, model_type=model_type, country=country)
            save_model(result, country=country)
            _cache["models"].pop(country, None)

            response = {
                "status": "success",
                "country": country,
                "model_type": model_type,
                "rmse": round(result["metrics"]["rmse"], 2),
                "mae": round(result["metrics"]["mae"], 2),
                "runtime": round(result["metadata"]["runtime_seconds"], 2),
            }

        log_general(f"Training completed for {country}")
        return jsonify(response)

    except Exception as e:
        log_general(f"Training error: {str(e)}", level="ERROR")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/logs/<log_type>", methods=["GET"])
def logs_endpoint(log_type):
    """Retrieve log entries."""
    valid_types = ["train", "predict", "monitor", "general"]
    if log_type not in valid_types:
        return jsonify({"error": f"Invalid log type. Valid: {valid_types}"}), 400

    n = request.args.get("n", 20, type=int)
    entries = read_logs(log_type, n=n)
    return jsonify({"log_type": log_type, "count": len(entries), "entries": entries})


@app.route("/monitor", methods=["GET"])
def monitor_endpoint():
    """Get performance monitoring summary."""
    summary = get_monitoring_summary()
    return jsonify(summary)


@app.route("/countries", methods=["GET"])
def countries_endpoint():
    """List available countries."""
    _ensure_data()
    return jsonify({"countries": ["all"] + _cache["countries"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    print(f"Starting AAVAIL Revenue Prediction API on port {port}...")
    log_general("API server starting")
    app.run(host="0.0.0.0", port=port, debug=False)
