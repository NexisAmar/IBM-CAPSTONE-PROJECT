"""
Data ingestion module for the AAVAIL revenue prediction pipeline.

Provides automated functions to:
  - Load raw CSV transaction data
  - Clean and preprocess records
  - Aggregate into daily/monthly time-series by country
  - Provide a unified ingestion pipeline callable from scripts or API
"""

import os
import re
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cs-train")


def load_raw_data(data_dir=None):
    """
    Load all CSV files from the cs-train directory.
    Returns a single concatenated DataFrame.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in sorted(csv_files):
        filepath = os.path.join(data_dir, f)
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total raw records: {len(combined)}")
    return combined


def clean_data(df):
    """
    Clean the raw transaction data:
      - Remove cancelled invoices (InvoiceNo starting with 'C')
      - Drop rows with missing CustomerID
      - Remove rows with non-positive Quantity or UnitPrice
      - Parse InvoiceDate to datetime
      - Ensure Revenue column exists
    """
    original_len = len(df)

    # Remove cancelled invoices
    if "InvoiceNo" in df.columns:
        mask = df["InvoiceNo"].astype(str).str.startswith("C")
        df = df[~mask].copy()

    # Drop missing customer IDs
    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"])

    # Remove non-positive quantities and prices
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    # Parse dates
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])

    # Ensure Revenue column
    if "Revenue" not in df.columns and "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df = df.reset_index(drop=True)
    print(f"  Cleaned: {original_len} -> {len(df)} records ({original_len - len(df)} removed)")
    return df


def aggregate_revenue(df, freq="D", country=None):
    """
    Aggregate transaction data into a time-series of revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data with 'InvoiceDate', 'Revenue', 'Country' columns.
    freq : str
        Resampling frequency: 'D' for daily, 'M' for monthly.
    country : str or None
        If specified, filter to a specific country. If None, aggregate all countries.

    Returns
    -------
    pd.DataFrame
        Time-series DataFrame with 'date' and 'revenue' columns.
    """
    if country and country != "all":
        df = df[df["Country"] == country].copy()

    ts = df.set_index("InvoiceDate").resample(freq)["Revenue"].sum().reset_index()
    ts.columns = ["date", "revenue"]
    ts = ts.sort_values("date").reset_index(drop=True)

    # Fill missing dates with 0 revenue
    full_range = pd.date_range(start=ts["date"].min(), end=ts["date"].max(), freq=freq)
    ts = ts.set_index("date").reindex(full_range, fill_value=0.0).reset_index()
    ts.columns = ["date", "revenue"]

    return ts


def get_available_countries(df):
    """Return sorted list of unique countries in the dataset."""
    if "Country" in df.columns:
        return sorted(df["Country"].unique().tolist())
    return []


def ingest_data(data_dir=None, freq="D", country=None):
    """
    Full ingestion pipeline: load -> clean -> aggregate.
    
    This is the main entry point for automated data ingestion.

    Parameters
    ----------
    data_dir : str
        Path to directory containing raw CSV files.
    freq : str
        Time-series frequency ('D' or 'M').
    country : str or None
        Country filter, or None / 'all' for combined.

    Returns
    -------
    tuple
        (time_series_df, countries_list)
    """
    print("=== Data Ingestion Pipeline ===")
    df = load_raw_data(data_dir)
    df = clean_data(df)
    countries = get_available_countries(df)
    ts = aggregate_revenue(df, freq=freq, country=country)
    print(f"  Time-series shape: {ts.shape}")
    print(f"  Date range: {ts['date'].min()} to {ts['date'].max()}")
    print(f"  Available countries: {countries}")
    return ts, countries


if __name__ == "__main__":
    ts, countries = ingest_data()
    print(f"\nSample data:\n{ts.head(10)}")
    print(f"\nCountries: {countries}")
