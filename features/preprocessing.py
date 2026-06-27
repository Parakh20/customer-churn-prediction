"""Feature preprocessing pipeline for customer churn prediction.

Provides:
- split_data: stratified 70/15/15 train/val/test split
- build_preprocessor: unfitted ColumnTransformer (OrdinalEncoder / OneHotEncoder / StandardScaler)
- fit_transform_splits: fit on train only, transform all three splits (no leakage)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_SEED = 42

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ordinal column specifications
# ---------------------------------------------------------------------------

# Maps ordinal column name → ordered category list (ascending)
_ORDINAL_SPECS: dict[str, list[str]] = {
    "tenure_bucket": ["early", "mid", "loyal"],
}

# Raw Telco categorical columns (object dtype) that are not ordinal
_KNOWN_NOMINAL_COLS: frozenset[str] = frozenset(
    {
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_columns(
    feature_names: list[str],
    X_ref: pd.DataFrame | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Partition feature_names into (ordinal, nominal, numeric) groups.

    If X_ref is supplied its dtypes are used for the ordinal/nominal/numeric
    split; otherwise falls back to hardcoded knowledge of the Telco dataset.

    Args:
        feature_names: Ordered list of feature column names to classify.
        X_ref: Optional reference DataFrame (e.g. X_train) used to infer dtypes.

    Returns:
        Tuple of (ordinal_cols, nominal_cols, numeric_cols).
    """
    ordinal_cols = [f for f in feature_names if f in _ORDINAL_SPECS]
    remaining = [f for f in feature_names if f not in _ORDINAL_SPECS]

    if X_ref is not None:
        nominal_cols = [
            f
            for f in remaining
            if X_ref[f].dtype == object or str(X_ref[f].dtype) == "category"
        ]
    else:
        nominal_cols = [f for f in remaining if f in _KNOWN_NOMINAL_COLS]

    numeric_cols = [f for f in remaining if f not in nominal_cols]

    return ordinal_cols, nominal_cols, numeric_cols


def _build_ordinal_pipeline(ordinal_cols: list[str]) -> Pipeline:
    """Return an impute→OrdinalEncoder pipeline for ordinal columns."""
    categories = [_ORDINAL_SPECS[col] for col in ordinal_cols]
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def _build_nominal_pipeline() -> Pipeline:
    """Return an impute→OneHotEncoder pipeline for nominal columns."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def _build_numeric_pipeline() -> Pipeline:
    """Return an impute→StandardScaler pipeline for numeric columns."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_preprocessor(
    feature_names: list[str],
    X_ref: pd.DataFrame | None = None,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the given feature columns.

    Callers must call .fit() (or .fit_transform()) on training data only to
    avoid leakage into val/test sets.

    Column groups:
    - Ordinals  → OrdinalEncoder (with known orderings)
    - Nominals  → OneHotEncoder (handle_unknown='ignore', dense)
    - Numerics  → StandardScaler

    Each group also includes a SimpleImputer to handle NaN values safely.

    Args:
        feature_names: List of feature column names the transformer will see.
        X_ref: Optional reference DataFrame used to infer column dtypes.
               If omitted, falls back to hardcoded Telco column knowledge.

    Returns:
        Unfitted ColumnTransformer ready to be fit on training data.
    """
    if not feature_names:
        raise ValueError("feature_names must be non-empty.")

    ordinal_cols, nominal_cols, numeric_cols = _classify_columns(feature_names, X_ref)

    log.info(
        "Preprocessor column groups — ordinal: %d, nominal: %d, numeric: %d",
        len(ordinal_cols),
        len(nominal_cols),
        len(numeric_cols),
    )

    transformers: list[tuple] = []

    if ordinal_cols:
        transformers.append(("ordinal", _build_ordinal_pipeline(ordinal_cols), ordinal_cols))

    if nominal_cols:
        transformers.append(("nominal", _build_nominal_pipeline(), nominal_cols))

    if numeric_cols:
        transformers.append(("numeric", _build_numeric_pipeline(), numeric_cols))

    if not transformers:
        raise ValueError("No transformers could be built from the supplied feature names.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Stratified 70/15/15 train/val/test split.

    The target column is binarised (Yes→1, No→0) before splitting.
    Stratification is applied at every split step to preserve class balance.

    Args:
        df: Full engineered DataFrame including the target column.
        target_col: Name of the binary target column (default "Churn").
        test_size: Fraction of total data reserved for the test set.
        val_size: Fraction of total data reserved for the validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y_raw = df[target_col]
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0}).astype(int)
    else:
        y = y_raw.astype(int)

    X = df.drop(columns=[target_col])

    # Step 1: split off test set (15% of total)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: split remaining 85% into train / val
    # val_size_of_remaining so that val ends up as val_size fraction of total
    val_size_of_remaining = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_of_remaining,
        stratify=y_temp,
        random_state=random_state,
    )

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_transform_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """Fit preprocessor on X_train and transform all three splits.

    The preprocessor is fitted exclusively on training data to prevent
    leakage into validation and test sets.

    Args:
        X_train: Training feature DataFrame.
        X_val: Validation feature DataFrame.
        X_test: Test feature DataFrame.
        feature_names: Feature columns to include (subset of DataFrame columns).

    Returns:
        Tuple of (X_train_arr, X_val_arr, X_test_arr, fitted_preprocessor).
    """
    preprocessor = build_preprocessor(feature_names, X_ref=X_train[feature_names])

    X_train_arr = preprocessor.fit_transform(X_train[feature_names])
    X_val_arr = preprocessor.transform(X_val[feature_names])
    X_test_arr = preprocessor.transform(X_test[feature_names])

    log.info(
        "Transformed shapes — train: %s, val: %s, test: %s",
        X_train_arr.shape,
        X_val_arr.shape,
        X_test_arr.shape,
    )

    return X_train_arr, X_val_arr, X_test_arr, preprocessor


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")

    print(f"Loading dataset from {csv_path} ...")
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    total_rows = len(raw)

    sys.path.insert(0, project_root)
    from features.engineering import engineer  # noqa: E402
    from features.selection import select_features  # noqa: E402

    print("Running engineer() ...")
    engineered = engineer(raw)

    print("Running select_features() ...")
    selected_features, _ = select_features(engineered, target_col="Churn")
    print(f"  Selected {len(selected_features)} features: {selected_features}")

    print("Splitting data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        engineered, target_col="Churn"
    )

    # --- Assert split ratios within ±1% ---
    train_ratio = len(X_train) / total_rows
    val_ratio = len(X_val) / total_rows
    test_ratio = len(X_test) / total_rows

    print(
        f"  Split ratios — train: {train_ratio:.3f}, val: {val_ratio:.3f}, test: {test_ratio:.3f}"
    )

    tolerance = 0.01
    assert abs(train_ratio - 0.70) <= tolerance, (
        f"FAIL — train ratio {train_ratio:.3f} not within ±1% of 0.70"
    )
    assert abs(val_ratio - 0.15) <= tolerance, (
        f"FAIL — val ratio {val_ratio:.3f} not within ±1% of 0.15"
    )
    assert abs(test_ratio - 0.15) <= tolerance, (
        f"FAIL — test ratio {test_ratio:.3f} not within ±1% of 0.15"
    )
    print("  Split ratio assertions passed.")

    print("Fitting preprocessor on train data and transforming all splits ...")
    X_train_arr, X_val_arr, X_test_arr, fitted_preprocessor = fit_transform_splits(
        X_train, X_val, X_test, selected_features
    )

    print(f"  X_train shape: {X_train_arr.shape}")
    print(f"  X_val   shape: {X_val_arr.shape}")
    print(f"  X_test  shape: {X_test_arr.shape}")

    # --- Assert no NaN in transformed arrays ---
    assert not np.isnan(X_train_arr).any(), "FAIL — NaN found in X_train_arr"
    assert not np.isnan(X_val_arr).any(), "FAIL — NaN found in X_val_arr"
    assert not np.isnan(X_test_arr).any(), "FAIL — NaN found in X_test_arr"
    print("  NaN check passed (no NaN in any transformed array).")

    # --- Assert no leakage: preprocessor fit on train, not on val/test ---
    # StandardScaler records n_samples_seen_ at fit time; verify it matches
    # the training set size (not val+test combined, which would indicate leakage).
    numeric_pipe = fitted_preprocessor.named_transformers_.get("numeric")
    if numeric_pipe is not None:
        n_seen = numeric_pipe.named_steps["scaler"].n_samples_seen_
        assert n_seen == len(X_train), (
            f"FAIL — scaler saw {n_seen} samples during fit, "
            f"but X_train has {len(X_train)} rows (leakage suspected)"
        )
        print(f"  Leakage check passed (scaler fit on {n_seen} train rows).")

    print("\nfeatures.preprocessing smoke test: OK")
