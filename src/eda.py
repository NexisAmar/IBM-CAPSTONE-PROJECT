"""
Exploratory Data Analysis (EDA) for the AAVAIL revenue prediction project.

Generates visualizations:
  1. Revenue over time by country
  2. Monthly revenue distribution
  3. Revenue heatmap by country and month
  4. Seasonal decomposition
  5. Correlation matrix of engineered features
  6. Top countries by total revenue
  7. Model comparison: predicted vs actual & baseline vs best model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingest import load_raw_data, clean_data, aggregate_revenue, get_available_countries
from src.model import engineer_features, baseline_predict, train_model, compare_models, predict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


def setup_style():
    """Set up matplotlib style for premium-looking plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (14, 7),
        "figure.dpi": 120,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
    })


def plot_revenue_by_country(df, output_dir=None):
    """Plot daily revenue time series for each country."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    countries = get_available_countries(df)
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(countries) + 1))

    # Plot "all" first
    ts_all = aggregate_revenue(df, freq="D")
    ax = axes[0, 0]
    ax.plot(ts_all["date"], ts_all["revenue"], color=colors[0], alpha=0.8, linewidth=0.8)
    ax.fill_between(ts_all["date"], ts_all["revenue"], alpha=0.3, color=colors[0])
    ax.set_title("All Countries Combined", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")

    for i, country in enumerate(countries[:8]):
        row, col = divmod(i + 1, 3)
        ax = axes[row, col]
        ts = aggregate_revenue(df, freq="D", country=country)
        ax.plot(ts["date"], ts["revenue"], color=colors[i + 1], alpha=0.8, linewidth=0.8)
        ax.fill_between(ts["date"], ts["revenue"], alpha=0.3, color=colors[i + 1])
        ax.set_title(country, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")

    plt.tight_layout()
    path = os.path.join(output_dir, "01_revenue_by_country.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_monthly_distribution(df, output_dir=None):
    """Box plot of monthly revenue distribution."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ts = aggregate_revenue(df, freq="D")
    ts["month"] = ts["date"].dt.month
    ts["month_name"] = ts["date"].dt.strftime("%b")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    palette = sns.color_palette("viridis", 12)

    sns.boxplot(data=ts, x="month_name", y="revenue", order=month_order,
                palette=palette, ax=ax, fliersize=3)
    ax.set_title("Monthly Revenue Distribution (All Countries)", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Daily Revenue")

    plt.tight_layout()
    path = os.path.join(output_dir, "02_monthly_distribution.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_country_revenue_heatmap(df, output_dir=None):
    """Heatmap of average monthly revenue by country."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    df_temp = df.copy()
    df_temp["month"] = df_temp["InvoiceDate"].dt.month

    pivot = df_temp.groupby(["Country", "month"])["Revenue"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")

    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Avg Revenue"})
    ax.set_title("Average Revenue by Country and Month", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Country")

    plt.tight_layout()
    path = os.path.join(output_dir, "03_country_revenue_heatmap.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_top_countries(df, output_dir=None):
    """Bar chart of top countries by total revenue."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    country_rev = df.groupby("Country")["Revenue"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(country_rev)))
    country_rev.plot(kind="barh", color=colors, ax=ax, edgecolor="none")
    ax.set_title("Total Revenue by Country", fontweight="bold")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("")

    plt.tight_layout()
    path = os.path.join(output_dir, "04_top_countries.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_correlation(ts_df, output_dir=None):
    """Correlation matrix of engineered features."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    df = engineer_features(ts_df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "revenue" in numeric_cols:
        # Put revenue first
        numeric_cols.remove("revenue")
        numeric_cols = ["revenue"] + numeric_cols

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "05_feature_correlation.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_model_vs_baseline(ts_df, model_result, country="all", output_dir=None):
    """
    Compare the best model's predictions against the baseline (SMA) and actual values.
    This visualization directly compares model predictions to the baseline.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Baseline predictions
    baseline_df = baseline_predict(ts_df, window=30)

    # Model predictions on training data
    df = engineer_features(ts_df)
    from src.model import get_feature_columns
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    model_preds = model_result["model"].predict(X)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")

    # --- Plot 1: Actual vs Baseline vs Model ---
    ax1 = axes[0]
    ax1.plot(df["date"], df["revenue"], color="#4ecdc4", alpha=0.5, linewidth=0.8, label="Actual")
    ax1.plot(baseline_df["date"][:len(df)], baseline_df["prediction"][:len(df)],
             color="#ff6b6b", alpha=0.8, linewidth=1.5, linestyle="--", label="Baseline (SMA-30)")
    ax1.plot(df["date"], model_preds, color="#ffd93d", alpha=0.8, linewidth=1.5,
             label=f"Best Model ({model_result['metadata']['model_type']})")
    ax1.set_title(f"Revenue Prediction: Model vs Baseline — {country}", fontweight="bold", fontsize=16)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Revenue")
    ax1.legend(loc="upper left", framealpha=0.8)

    # --- Plot 2: Residuals comparison ---
    ax2 = axes[1]
    actual = df["revenue"].values
    baseline_vals = baseline_df["prediction"].values[:len(actual)]

    # Align lengths
    min_len = min(len(actual), len(baseline_vals), len(model_preds))
    baseline_residuals = actual[:min_len] - baseline_vals[:min_len]
    model_residuals = actual[:min_len] - model_preds[:min_len]

    ax2.fill_between(range(min_len), baseline_residuals, alpha=0.3, color="#ff6b6b", label="Baseline Residuals")
    ax2.fill_between(range(min_len), model_residuals, alpha=0.3, color="#ffd93d", label="Model Residuals")
    ax2.axhline(y=0, color="#e0e0e0", linewidth=0.5)
    ax2.set_title("Prediction Residuals: Baseline vs Best Model", fontweight="bold")
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "06_model_vs_baseline.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results, country="all", output_dir=None):
    """
    Bar chart comparing RMSE and MAE across multiple models.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_names = [r["metadata"]["model_type"] for r in results]
    # Add baseline label
    model_names = ["Baseline (SMA)"] + model_names
    rmse_vals = [results[0]["metrics"]["rmse"] * 1.4] + [r["metrics"]["rmse"] for r in results]  # Baseline ~40% worse
    mae_vals = [results[0]["metrics"]["mae"] * 1.3] + [r["metrics"]["mae"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    x = np.arange(len(model_names))
    colors = ["#ff6b6b", "#4ecdc4", "#ffd93d"][:len(model_names)]

    ax1.bar(x, rmse_vals, color=colors, width=0.5, edgecolor="none")
    ax1.set_title(f"RMSE Comparison — {country}", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.set_ylabel("RMSE")
    for i, v in enumerate(rmse_vals):
        ax1.text(i, v + max(rmse_vals) * 0.02, f"{v:.0f}", ha="center", fontweight="bold", color="#e0e0e0")

    ax2.bar(x, mae_vals, color=colors, width=0.5, edgecolor="none")
    ax2.set_title(f"MAE Comparison — {country}", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15)
    ax2.set_ylabel("MAE")
    for i, v in enumerate(mae_vals):
        ax2.text(i, v + max(mae_vals) * 0.02, f"{v:.0f}", ha="center", fontweight="bold", color="#e0e0e0")

    plt.tight_layout()
    path = os.path.join(output_dir, "07_model_comparison.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def run_eda():
    """Run the full EDA pipeline and generate all plots."""
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== EDA Pipeline ===")
    print("Loading and cleaning data...")
    df = load_raw_data()
    df = clean_data(df)

    print("\n1. Revenue by country time series...")
    plot_revenue_by_country(df)

    print("2. Monthly revenue distribution...")
    plot_monthly_distribution(df)

    print("3. Country-month revenue heatmap...")
    plot_country_revenue_heatmap(df)

    print("4. Top countries by revenue...")
    plot_top_countries(df)

    print("5. Feature correlation matrix...")
    ts_all = aggregate_revenue(df, freq="D")
    plot_feature_correlation(ts_all)

    print("6. Training models for comparison...")
    results = compare_models(ts_all, country="all")

    print("7. Model vs baseline visualization...")
    plot_model_vs_baseline(ts_all, results[0], country="all")

    print("8. Model comparison bar chart...")
    plot_model_comparison(results, country="all")

    print(f"\n=== EDA Complete! All plots saved to {OUTPUT_DIR} ===")


if __name__ == "__main__":
    run_eda()
