"""Exploratory Data Analysis for Cookie Cats A/B test."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import load_data, create_binary_treatment


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def summarize_data(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    summary = {
        "n_users": len(df),
        "n_control": (df["version"] == "gate_30").sum(),
        "n_treatment": (df["version"] == "gate_40").sum(),
        "retention_1d_overall": df["retention_1"].mean(),
        "retention_7d_overall": df["retention_7"].mean(),
        "median_rounds": df["sum_gamerounds"].median(),
        "mean_rounds": df["sum_gamerounds"].mean(),
    }
    return summary


def check_randomization_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Check covariate balance between treatment and control."""
    df = create_binary_treatment(df)

    balance = df.groupby("treatment").agg({
        "sum_gamerounds": ["mean", "std", "median"],
        "retention_1": "mean",
        "retention_7": "mean",
        "userid": "count"
    }).round(4)

    return balance


def plot_gamerounds_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Plot distribution of game rounds by treatment group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram (capped at 99th percentile for visibility)
    cap = df["sum_gamerounds"].quantile(0.99)
    df_capped = df[df["sum_gamerounds"] <= cap]

    for version, color in [("gate_30", "blue"), ("gate_40", "orange")]:
        data = df_capped[df_capped["version"] == version]["sum_gamerounds"]
        axes[0].hist(data, bins=50, alpha=0.5, label=version, color=color)

    axes[0].set_xlabel("Game Rounds")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Game Rounds (capped at 99th percentile)")
    axes[0].legend()

    # Box plot
    sns.boxplot(data=df_capped, x="version", y="sum_gamerounds", ax=axes[1])
    axes[1].set_title("Game Rounds by Treatment Group")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "gamerounds_distribution.png", dpi=150)
    plt.show()


def plot_retention_rates(df: pd.DataFrame, save: bool = True) -> None:
    """Plot retention rates by treatment group."""
    retention = df.groupby("version")[["retention_1", "retention_7"]].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    retention.plot(kind="bar", ax=ax)
    ax.set_ylabel("Retention Rate")
    ax.set_title("Retention Rates by Treatment Group")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(["1-Day Retention", "7-Day Retention"])

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "retention_rates.png", dpi=150)
    plt.show()


def run_eda() -> dict:
    """Run complete EDA pipeline."""
    print("Loading data...")
    df = load_data()

    print("\n=== Data Summary ===")
    summary = summarize_data(df)
    for k, v in summary.items():
        print(f"{k}: {v:,.4f}" if isinstance(v, float) else f"{k}: {v:,}")

    print("\n=== Randomization Balance Check ===")
    balance = check_randomization_balance(df)
    print(balance)

    print("\n=== Generating Plots ===")
    plot_gamerounds_distribution(df)
    plot_retention_rates(df)

    return {"summary": summary, "balance": balance}


if __name__ == "__main__":
    run_eda()
