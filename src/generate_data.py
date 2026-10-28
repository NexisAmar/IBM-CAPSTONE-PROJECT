"""
Generate synthetic AAVAIL cs-train dataset for the capstone project.
Simulates daily revenue data across multiple countries with realistic
seasonal patterns, trends, and noise.
"""

import os
import numpy as np
import pandas as pd

COUNTRIES = [
    "United Kingdom", "Germany", "France", "Portugal",
    "EIRE", "Spain", "Netherlands", "Norway"
]

STOCK_CODES = [f"SC{str(i).zfill(4)}" for i in range(1, 51)]


def generate_country_data(country, start_date="2017-11-01", end_date="2019-07-31", seed=None):
    """Generate synthetic invoice data for a single country."""
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    # Country-specific revenue multiplier
    multipliers = {
        "United Kingdom": 5.0, "Germany": 1.5, "France": 1.3,
        "Portugal": 0.6, "EIRE": 0.8, "Spain": 0.7,
        "Netherlands": 0.9, "Norway": 0.5
    }
    mult = multipliers.get(country, 1.0)

    for date in dates:
        # Number of invoices per day (varies by country size)
        n_invoices = max(1, int(np.random.poisson(lam=10 * mult)))
        for _ in range(n_invoices):
            invoice_no = f"INV{np.random.randint(100000, 999999)}"
            stock_code = np.random.choice(STOCK_CODES)
            quantity = np.random.randint(1, 25)

            # Seasonal component: higher revenue in Q4 (holiday season)
            month = date.month
            seasonal = 1.0
            if month in [11, 12]:
                seasonal = 1.5
            elif month in [1, 2]:
                seasonal = 0.8
            elif month in [6, 7, 8]:
                seasonal = 1.1

            # Trend: slight upward trend over time
            days_from_start = (date - dates[0]).days
            trend = 1.0 + 0.0003 * days_from_start

            unit_price = round(np.random.lognormal(mean=1.5, sigma=0.8) * seasonal * trend, 2)
            revenue = round(quantity * unit_price, 2)

            records.append({
                "InvoiceNo": invoice_no,
                "StockCode": stock_code,
                "Description": f"Product {stock_code}",
                "Quantity": quantity,
                "InvoiceDate": date.strftime("%Y-%m-%d %H:%M"),
                "UnitPrice": unit_price,
                "CustomerID": np.random.randint(10000, 99999),
                "Country": country,
                "Revenue": revenue,
            })

    return records


def generate_dataset(output_dir="data/cs-train"):
    """Generate the full cs-train dataset."""
    os.makedirs(output_dir, exist_ok=True)

    all_records = []
    for i, country in enumerate(COUNTRIES):
        print(f"  Generating data for {country}...")
        records = generate_country_data(country, seed=42 + i)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = os.path.join(output_dir, "ts-data.csv")
    df.to_csv(output_path, index=False)
    print(f"  Dataset saved to {output_path} ({len(df)} rows)")
    return output_path


if __name__ == "__main__":
    print("Generating AAVAIL cs-train dataset...")
    generate_dataset()
    print("Done.")
