"""Utility functions for data loading and common operations."""

import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def load_data() -> pd.DataFrame:
    """Load Cookie Cats dataset."""
    filepath = DATA_DIR / "cookie_cats.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run download_data() first."
        )
    return pd.read_csv(filepath)


def download_data() -> None:
    """Download Cookie Cats dataset from Kaggle."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("mursideyarkin/mobile-games-ab-testing-cookie-cats")
        print(f"Dataset downloaded to: {path}")

        # Copy to data directory
        import shutil
        DATA_DIR.mkdir(exist_ok=True)
        for f in Path(path).glob("*.csv"):
            shutil.copy(f, DATA_DIR / "cookie_cats.csv")
            print(f"Copied to {DATA_DIR / 'cookie_cats.csv'}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Manual download: https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats")


def create_binary_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Convert version column to binary treatment indicator."""
    df = df.copy()
    df["treatment"] = (df["version"] == "gate_40").astype(int)
    return df


if __name__ == "__main__":
    download_data()
