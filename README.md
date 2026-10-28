# AAVAIL Revenue Prediction — IBM AI Enterprise Workflow Capstone

A production-grade machine learning pipeline for predicting daily revenue across multiple countries for the AAVAIL streaming platform.

## Project Structure

```
IBM-CAPSTONE-PROJECT/
├── app.py                  # Flask API server
├── run_tests.py            # Single script to run all unit tests
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── data/
│   └── cs-train/           # Raw transaction data (CSV)
├── models/                 # Trained model files (.joblib)
│   └── test/               # Isolated test model directory
├── logs/                   # Production log files (JSON-lines)
│   └── test/               # Isolated test log directory
├── src/
│   ├── __init__.py
│   ├── generate_data.py    # Synthetic dataset generator
│   ├── ingest.py           # Data ingestion & preprocessing pipeline
│   ├── model.py            # Model training, comparison, prediction
│   ├── logger.py           # Structured logging (train/predict/monitor)
│   ├── monitor.py          # Performance monitoring (Wasserstein drift)
│   └── eda.py              # Exploratory Data Analysis with visualizations
├── static/                 # EDA visualization outputs (PNG)
└── tests/
    ├── __init__.py
    ├── test_api.py          # API endpoint tests
    ├── test_model.py        # Model training & prediction tests
    └── test_logger.py       # Logging module tests
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python -m src.generate_data
```

### 3. Run EDA & Train Models
```bash
python -m src.eda
```

### 4. Run All Unit Tests
```bash
python run_tests.py
```

### 5. Start the API
```bash
python app.py
```

### 6. Docker
```bash
docker build -t aavail-revenue-api .
docker run -p 80:80 aavail-revenue-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Get revenue prediction (`country`, `date`) |
| POST | `/train` | Train models (`country`, `model_type`) |
| GET | `/logs/<type>` | Retrieve logs (train/predict/monitor/general) |
| GET | `/monitor` | Performance monitoring summary |
| GET | `/countries` | List available countries |

### Example API Calls

```bash
# Predict for all countries
curl -X POST "http://localhost/predict" -H "Content-Type: application/json" \
  -d '{"country": "all", "date": "2019-08-01"}'

# Predict for a specific country
curl -X POST "http://localhost/predict" -H "Content-Type: application/json" \
  -d '{"country": "United Kingdom", "date": "2019-08-01"}'

# Train models (compare RF vs GB)
curl -X POST "http://localhost/train" -H "Content-Type: application/json" \
  -d '{"country": "all", "model_type": "compare"}'
```

## Key Features

- **Automated Data Ingestion**: `src/ingest.py` — load, clean, aggregate as reusable functions
- **Multiple Model Comparison**: Random Forest vs Gradient Boosting with time-series CV
- **Baseline Comparison**: SMA baseline vs ML models with visual comparison
- **Performance Monitoring**: Wasserstein distance drift detection, error tracking
- **Structured Logging**: JSON-lines format for train/predict/monitor events
- **Test Isolation**: Unit tests use temp directories — never touch production models/logs
- **EDA Visualizations**: 7+ plots including revenue trends, heatmaps, model vs baseline
- **Docker Containerization**: Full pipeline in a reproducible container
