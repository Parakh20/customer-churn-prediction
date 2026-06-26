"""Feature engineering module for customer churn prediction.

All transformations are pure functions — input DataFrames are never mutated.
The Churn column is never used as a feature input.
"""

from __future__ import annotations

import pandas as pd

RANDOM_SEED = 42

# Service columns used to count active services
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_AUTO_PAYMENT_METHODS = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


def _add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenure-based derived features."""
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, float("inf")],
        labels=["early", "mid", "loyal"],
    ).astype(str)

    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["tenure_squared"] = df["tenure"] ** 2

    return df


def _add_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add service-count and bundle derived features."""
    df = df.copy()

    service_flags = df[_SERVICE_COLS].eq("Yes")
    df["num_services"] = service_flags.sum(axis=1)

    df["has_support"] = (
        (df["TechSupport"] == "Yes") | (df["OnlineSecurity"] == "Yes")
    ).astype(int)

    df["is_bundle"] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)

    return df


def _add_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spend-ratio and charge-proxy derived features.

    Requires num_services to already be present in df.
    NaN in TotalCharges propagates naturally without dropping rows.
    """
    df = df.copy()

    df["monthly_charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    df["total_vs_expected"] = df["TotalCharges"] / (
        df["tenure"] * df["MonthlyCharges"] + 1e-9
    )

    df["charge_increase_proxy"] = (
        df["TotalCharges"] / (df["tenure"] + 1) - df["MonthlyCharges"]
    )

    return df


def _add_contract_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contract and payment risk derived features.

    Requires is_new_customer to already be present in df.
    """
    df = df.copy()

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    df["payment_auto"] = df["PaymentMethod"].isin(_AUTO_PAYMENT_METHODS).astype(int)

    df["risk_score"] = (
        df["is_month_to_month"] * 2
        + df["is_new_customer"] * 1.5
        - df["payment_auto"] * 1
    )

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction term features.

    Requires tenure, MonthlyCharges, is_month_to_month, is_new_customer.
    """
    df = df.copy()

    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["mtm_x_new"] = df["is_month_to_month"] * df["is_new_customer"]

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with all derived features appended.

    The Churn column is never read or used as a feature input.
    Input df is never mutated.

    Args:
        df: Raw Telco churn DataFrame (must include tenure, MonthlyCharges,
            TotalCharges, Contract, PaymentMethod, and the service columns).

    Returns:
        New DataFrame with original columns plus all engineered feature columns.
    """
    result = df.copy()

    result = _add_tenure_features(result)
    result = _add_service_features(result)
    result = _add_spend_features(result)       # needs num_services
    result = _add_contract_risk_features(result)  # needs is_new_customer
    result = _add_interaction_features(result)    # needs is_month_to_month, is_new_customer

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_NEW_COLUMNS = [
    # tenure
    "tenure_bucket",
    "is_new_customer",
    "tenure_squared",
    # service
    "num_services",
    "has_support",
    "is_bundle",
    # spend
    "monthly_charge_per_service",
    "total_vs_expected",
    "charge_increase_proxy",
    # contract risk
    "is_month_to_month",
    "payment_auto",
    "risk_score",
    # interactions
    "tenure_x_monthly",
    "mtm_x_new",
]

if __name__ == "__main__":
    import sys
    import os

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv"
    )
    csv_path = os.path.normpath(csv_path)

    print(f"Loading dataset from {csv_path} ...")
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    # Verify Churn is NOT passed to engineer as a feature — it stays in df but
    # engineer() must not read it.  We drop it from a copy to confirm the
    # function works without it.
    raw_no_churn = raw.drop(columns=["Churn"])

    print("Running engineer() without Churn column ...")
    engineered = engineer(raw_no_churn)

    # 1. All new columns exist
    missing = [c for c in _NEW_COLUMNS if c not in engineered.columns]
    if missing:
        print(f"FAIL — missing columns: {missing}")
        sys.exit(1)
    print(f"  All {len(_NEW_COLUMNS)} new columns present.")

    # 2. Input df not mutated
    assert "tenure_bucket" not in raw_no_churn.columns, "FAIL — input df was mutated"
    print("  Input DataFrame not mutated.")

    # 3. Churn not used as input (already enforced above)
    print("  Churn column not used as feature input.")

    # 4. No NaN explosion — allow up to 5 % NaN in any new column
    nan_threshold = 0.05 * len(engineered)
    nan_issues = []
    for col in _NEW_COLUMNS:
        n_nan = engineered[col].isna().sum()
        if n_nan > nan_threshold:
            nan_issues.append((col, n_nan, n_nan / len(engineered)))
    if nan_issues:
        print("FAIL — NaN explosion in columns:")
        for col, n, pct in nan_issues:
            print(f"  {col}: {n} NaNs ({pct:.1%})")
        sys.exit(1)
    print(f"  NaN check passed (threshold {nan_threshold:.0f} rows per column).")

    # 5. Sample values
    print("\nSample values (first 3 rows):")
    print(engineered[_NEW_COLUMNS].head(3).to_string())

    print("\nfeatures.engineering smoke test: OK")
