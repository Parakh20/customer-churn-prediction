"""Data download and preprocessing utilities for customer churn datasets."""

import os
from pathlib import Path
from typing import Optional
import io

import pandas as pd
import numpy as np
import requests

RANDOM_SEED = 42

# Dataset URLs and paths
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_DIR = Path(__file__).parent / "raw"
TELCO_PATH = RAW_DATA_DIR / "telco_churn.csv"
ECOMMERCE_PATH = RAW_DATA_DIR / "ecommerce_churn.csv"


def _ensure_raw_dir() -> None:
    """Ensure raw data directory exists."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_telco() -> pd.DataFrame:
    """Download IBM Telco dataset from GitHub."""
    try:
        response = requests.get(TELCO_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download Telco dataset: {e}") from e


def _preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Telco dataset: coerce TotalCharges, strip whitespace."""
    df = df.copy()

    # Coerce TotalCharges to numeric (blank → NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


def _validate_telco(df: pd.DataFrame) -> None:
    """Validate Telco dataset shape and churn rate."""
    expected_shape = (7043, 21)
    churn_col = "Churn"

    # Check shape
    if df.shape != expected_shape:
        raise ValueError(
            f"Telco dataset shape mismatch. Expected {expected_shape}, got {df.shape}"
        )

    # Check churn column exists
    if churn_col not in df.columns:
        raise ValueError(f"Churn column '{churn_col}' not found in dataset")

    # Calculate churn rate
    churn_rate = (df[churn_col] == "Yes").sum() / len(df)
    expected_range = (0.25, 0.28)
    if not (expected_range[0] <= churn_rate <= expected_range[1]):
        raise ValueError(
            f"Telco churn rate {churn_rate:.4f} outside expected range {expected_range}"
        )


def _try_kaggle_ecommerce() -> Optional[pd.DataFrame]:
    """Try to download e-commerce churn dataset from Kaggle."""
    try:
        import kaggle

        # Search for most-downloaded e-commerce churn dataset
        try:
            kaggle.api.dataset_download_files(
                "ecommerce-churn-data",  # Common dataset name
                path=str(RAW_DATA_DIR),
                unzip=True,
            )
            # Try to read the downloaded file
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if csv_files:
                # Use the first CSV found (typically the main dataset)
                for csv_file in csv_files:
                    if "ecommerce" in csv_file.name.lower() or "churn" in csv_file.name.lower():
                        df = pd.read_csv(csv_file)
                        csv_file.rename(ECOMMERCE_PATH)
                        return df
                # If no matching name, use the first CSV
                df = pd.read_csv(csv_files[0])
                csv_files[0].rename(ECOMMERCE_PATH)
                return df
        except Exception:
            # Kaggle API call failed, return None to trigger fallback
            return None
    except ImportError:
        # kaggle package not installed
        return None


def _create_synthetic_ecommerce() -> pd.DataFrame:
    """Create synthetic e-commerce churn dataset as fallback."""
    np.random.seed(RANDOM_SEED)

    n_samples = 5000
    df = pd.DataFrame(
        {
            "customer_id": range(1, n_samples + 1),
            "tenure_months": np.random.randint(1, 60, n_samples),
            "monthly_spend": np.random.gamma(2, 50, n_samples).round(2),
            "total_spend": np.random.gamma(3, 500, n_samples).round(2),
            "num_purchases": np.random.poisson(5, n_samples),
            "avg_order_value": np.random.gamma(2, 30, n_samples).round(2),
            "days_since_purchase": np.random.randint(0, 365, n_samples),
            "support_tickets": np.random.poisson(1, n_samples),
            "satisfaction_score": np.random.randint(1, 6, n_samples),
            "has_loyalty_program": np.random.choice(["Yes", "No"], n_samples),
            "churn": np.random.choice(["Yes", "No"], n_samples, p=[0.26, 0.74]),
        }
    )

    return df


def download_telco() -> pd.DataFrame:
    """Download and preprocess Telco dataset."""
    _ensure_raw_dir()

    df = _download_telco()
    df = _preprocess_telco(df)
    _validate_telco(df)
    df.to_csv(TELCO_PATH, index=False)

    return df


def download_ecommerce() -> tuple[pd.DataFrame, str]:
    """Download e-commerce dataset, trying Kaggle first, falling back to synthetic.

    Returns:
        Tuple of (DataFrame, path_used) where path_used is "kaggle" or "synthetic"
    """
    _ensure_raw_dir()

    # Try Kaggle
    df = _try_kaggle_ecommerce()
    if df is not None:
        print("Using Kaggle e-commerce dataset")
        df.to_csv(ECOMMERCE_PATH, index=False)
        return df, "kaggle"

    # Fallback: Create synthetic dataset
    print("Kaggle unavailable, using synthetic e-commerce dataset")
    df = _create_synthetic_ecommerce()
    df.to_csv(ECOMMERCE_PATH, index=False)
    return df, "synthetic"


def download_all() -> dict:
    """Download all datasets and return summary."""
    results = {}

    # Download Telco
    print("Downloading Telco dataset...")
    telco_df = download_telco()
    telco_churn_rate = (telco_df["Churn"] == "Yes").sum() / len(telco_df)
    results["telco"] = {
        "shape": telco_df.shape,
        "churn_rate": telco_churn_rate,
        "path": str(TELCO_PATH),
    }

    # Download E-commerce
    print("Downloading e-commerce dataset...")
    ecommerce_df, ecommerce_source = download_ecommerce()
    results["ecommerce"] = {
        "shape": ecommerce_df.shape,
        "source": ecommerce_source,
        "path": str(ECOMMERCE_PATH),
    }

    return results


if __name__ == "__main__":
    # Smoke test
    results = download_all()

    # Validate Telco dataset
    telco_shape = results["telco"]["shape"]
    telco_churn = results["telco"]["churn_rate"]
    assert telco_shape == (7043, 21), f"Expected shape (7043, 21), got {telco_shape}"
    assert 0.25 <= telco_churn <= 0.28, f"Expected churn rate in [0.25, 0.28], got {telco_churn:.4f}"

    # Verify files exist
    assert TELCO_PATH.exists(), f"Telco dataset not found at {TELCO_PATH}"
    assert ECOMMERCE_PATH.exists(), f"E-commerce dataset not found at {ECOMMERCE_PATH}"

    print("data.download smoke test: OK")
    print(f"Telco dataset: {telco_shape}, churn rate: {telco_churn:.4f}")
    print(f"E-commerce dataset: {results['ecommerce']['shape']}, source: {results['ecommerce']['source']}")

