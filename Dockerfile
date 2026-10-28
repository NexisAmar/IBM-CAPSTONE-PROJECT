# AAVAIL Revenue Prediction Service
# Multi-stage build for optimized image size

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app

# Generate synthetic dataset
RUN python -m src.generate_data

# Run unit tests during build to ensure correctness
RUN python run_tests.py

# Train models for all countries
RUN python -c "\
from src.ingest import load_raw_data, clean_data, aggregate_revenue, get_available_countries; \
from src.model import compare_models, save_model; \
df = load_raw_data(); \
df = clean_data(df); \
countries = get_available_countries(df); \
for country in ['all'] + countries: \
    print(f'Training models for {country}...'); \
    ts = aggregate_revenue(df, freq='D', country=country if country != 'all' else None); \
    results = compare_models(ts, country=country); \
    save_model(results[0], country=country); \
    print(f'  Best: {results[0][\"metadata\"][\"model_type\"]}')
"

# Run EDA to generate visualizations
RUN python -m src.eda

# Expose port 80
EXPOSE 80

# Set environment variables
ENV PORT=80
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/')" || exit 1

# Start the API server
CMD ["python", "app.py"]
