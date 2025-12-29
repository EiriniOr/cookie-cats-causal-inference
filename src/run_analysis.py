"""Main script to run the complete analysis pipeline."""

import json
from pathlib import Path
from datetime import datetime

from src.utils import download_data, load_data
from src.eda import run_eda
from src.classical_ab import run_classical_analysis
from src.causal_methods import run_causal_analysis
from src.hte_analysis import run_hte_analysis
from src.sensitivity import run_sensitivity_analysis


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    """Run complete analysis pipeline."""
    print("=" * 60)
    print("Cookie Cats A/B Test - Causal Inference Analysis")
    print("=" * 60)

    # Check if data exists, download if not
    try:
        df = load_data()
        print(f"Data loaded: {len(df):,} rows")
    except FileNotFoundError:
        print("Downloading data...")
        download_data()
        df = load_data()

    # Run all analyses
    results = {}

    print("\n" + "=" * 60)
    print("PHASE 1: Exploratory Data Analysis")
    print("=" * 60)
    results["eda"] = run_eda()

    print("\n" + "=" * 60)
    print("PHASE 2: Classical A/B Testing")
    print("=" * 60)
    results["classical"] = run_classical_analysis()

    print("\n" + "=" * 60)
    print("PHASE 3: Causal Inference Methods")
    print("=" * 60)
    results["causal"] = run_causal_analysis()

    print("\n" + "=" * 60)
    print("PHASE 4: Heterogeneous Treatment Effects")
    print("=" * 60)
    results["hte"] = run_hte_analysis()

    print("\n" + "=" * 60)
    print("PHASE 5: Sensitivity Analysis")
    print("=" * 60)
    results["sensitivity"] = run_sensitivity_analysis()

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Outputs saved to: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    main()
