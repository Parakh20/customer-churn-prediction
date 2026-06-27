"""Imbalance handling strategies for customer churn prediction.

Four strategies are exposed, each returning a uniform result dict:
  {
    "strategy": str,
    "X_train_resampled": np.ndarray,
    "y_train_resampled": np.ndarray,
    "threshold": float,
    "val_metrics": dict,          # precision, recall, f1, auc_roc, auc_pr
    # strategy 4 only adds:
    "threshold_f1": float,
    "threshold_business": float,
  }

SMOTE is applied to train data ONLY — val and test are never resampled.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_SEED = 42
_DEFAULT_THRESHOLD = 0.5
_THRESHOLD_GRID = np.linspace(0.1, 0.9, 81)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Validate shapes and types before fitting.

    Raises:
        TypeError: If any input is not np.ndarray.
        ValueError: If X arrays are not 2-D or X/y lengths don't match.
    """
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X_{name} must be np.ndarray, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y_{name} must be np.ndarray, got {type(y)}")
        if X.ndim != 2:
            raise ValueError(f"X_{name} must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X_{name} rows ({len(X)}) != y_{name} length ({len(y)})")


def _fit_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str | None = None,
) -> LogisticRegression:
    """Fit and return a LogisticRegression with the given class_weight."""
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def _compute_val_metrics(
    model: LogisticRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Return val precision, recall, F1, AUC-ROC, AUC-PR at the given threshold."""
    proba = model.predict_proba(X_val)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_val, proba)),
        "auc_pr": float(average_precision_score(y_val, proba)),
    }


def _business_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return -(5*FN + FP) — higher is better (minimise weighted misclassification).

    False negatives cost 5× more than false positives (churner not retained vs.
    unnecessary retention offer).
    """
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return -(5 * fn + fp)


def _best_threshold(
    proba_val: np.ndarray,
    y_val: np.ndarray,
    grid: np.ndarray = _THRESHOLD_GRID,
) -> tuple[float, float]:
    """Return (threshold_business, threshold_f1) by scanning `grid`.

    threshold_business maximises -(5*FN + FP).
    threshold_f1 maximises F1 score.
    """
    best_biz_score = -np.inf
    best_f1_score = -np.inf
    threshold_business = grid[0]
    threshold_f1 = grid[0]

    for t in grid:
        y_pred = (proba_val >= t).astype(int)

        biz = _business_score(y_val, y_pred)
        if biz > best_biz_score:
            best_biz_score = biz
            threshold_business = float(t)

        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        if f1 > best_f1_score:
            best_f1_score = f1
            threshold_f1 = float(t)

    return threshold_business, threshold_f1


# ---------------------------------------------------------------------------
# Public strategy functions
# ---------------------------------------------------------------------------


def strategy_none(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Strategy 1: no resampling, no class weighting, threshold=0.5."""
    _validate_arrays(X_train, y_train, X_val, y_val)
    model = _fit_lr(X_train, y_train, class_weight=None)
    threshold = _DEFAULT_THRESHOLD
    val_metrics = _compute_val_metrics(model, X_val, y_val, threshold)

    log.info("[strategy_none] val metrics: %s", val_metrics)

    return {
        "strategy": "none",
        "X_train_resampled": X_train,
        "y_train_resampled": y_train,
        "threshold": threshold,
        "val_metrics": val_metrics,
    }


def strategy_class_weight(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Strategy 2: class_weight='balanced', threshold=0.5."""
    _validate_arrays(X_train, y_train, X_val, y_val)
    model = _fit_lr(X_train, y_train, class_weight="balanced")
    threshold = _DEFAULT_THRESHOLD
    val_metrics = _compute_val_metrics(model, X_val, y_val, threshold)

    log.info("[strategy_class_weight] val metrics: %s", val_metrics)

    return {
        "strategy": "class_weight",
        "X_train_resampled": X_train,
        "y_train_resampled": y_train,
        "threshold": threshold,
        "val_metrics": val_metrics,
    }


def strategy_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Strategy 3: SMOTE on train only, then fit LR, threshold=0.5.

    SMOTE is NEVER applied to val or test data.
    """
    _validate_arrays(X_train, y_train, X_val, y_val)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    log.info(
        "[strategy_smote] train size: %d → %d after SMOTE",
        len(y_train),
        len(y_resampled),
    )

    model = _fit_lr(X_resampled, y_resampled, class_weight=None)
    threshold = _DEFAULT_THRESHOLD
    val_metrics = _compute_val_metrics(model, X_val, y_val, threshold)

    log.info("[strategy_smote] val metrics: %s", val_metrics)

    return {
        "strategy": "smote",
        "X_train_resampled": X_resampled,
        "y_train_resampled": y_resampled,
        "threshold": threshold,
        "val_metrics": val_metrics,
    }


def strategy_threshold_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Strategy 4: class_weight='balanced' + threshold tuned on val by business metric.

    Searches np.linspace(0.1, 0.9, 81).
    Returns the threshold that maximises -(5*FN + FP).
    Also records the F1-optimal threshold for reference.
    """
    _validate_arrays(X_train, y_train, X_val, y_val)
    model = _fit_lr(X_train, y_train, class_weight="balanced")
    proba_val = model.predict_proba(X_val)[:, 1]

    threshold_business, threshold_f1 = _best_threshold(proba_val, y_val)

    log.info(
        "[strategy_threshold_tuning] threshold_business=%.3f  threshold_f1=%.3f",
        threshold_business,
        threshold_f1,
    )

    val_metrics = _compute_val_metrics(model, X_val, y_val, threshold_business)

    log.info("[strategy_threshold_tuning] val metrics at business threshold: %s", val_metrics)

    return {
        "strategy": "threshold_tuning",
        "X_train_resampled": X_train,
        "y_train_resampled": y_train,
        "threshold": threshold_business,
        "threshold_business": threshold_business,
        "threshold_f1": threshold_f1,
        "val_metrics": val_metrics,
    }


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all four strategies and return a comparison DataFrame.

    Columns: strategy, precision, recall, f1, auc_roc, auc_pr, threshold.
    Also prints the table to stdout when verbose=True.

    Args:
        X_train: Preprocessed training features (numpy array).
        y_train: Training labels (numpy array).
        X_val:   Preprocessed validation features (numpy array).
        y_val:   Validation labels (numpy array).
        verbose: If True, print the comparison table to stdout (default True).

    Returns:
        DataFrame with one row per strategy.
    """
    results = [
        strategy_none(X_train, y_train, X_val, y_val),
        strategy_class_weight(X_train, y_train, X_val, y_val),
        strategy_smote(X_train, y_train, X_val, y_val),
        strategy_threshold_tuning(X_train, y_train, X_val, y_val),
    ]

    rows = []
    for r in results:
        row = {"strategy": r["strategy"], **r["val_metrics"], "threshold": r["threshold"]}
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=["strategy", "precision", "recall", "f1", "auc_roc", "auc_pr", "threshold"],
    )

    if verbose:
        print("\n=== Strategy Comparison (validation set) ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")
    sys.path.insert(0, project_root)

    import pandas as _pd

    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    print(f"Loading dataset from {csv_path} ...")
    raw = _pd.read_csv(csv_path)
    raw["TotalCharges"] = _pd.to_numeric(raw["TotalCharges"], errors="coerce")

    print("Engineering features ...")
    engineered = engineer(raw)

    print("Selecting features ...")
    selected_features, _ = select_features(engineered, target_col="Churn")
    print(f"  Selected {len(selected_features)} features.")

    print("Splitting data ...")
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(
        engineered, target_col="Churn"
    )

    print("Preprocessing splits ...")
    X_train_arr, X_val_arr, X_test_arr, _ = fit_transform_splits(
        X_train_df, X_val_df, X_test_df, selected_features
    )

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    print("Running compare_strategies() ...")
    df_results = compare_strategies(X_train_arr, y_train_arr, X_val_arr, y_val_arr)

    # --- Assertions ---
    assert len(df_results) == 4, f"FAIL — expected 4 rows, got {len(df_results)}"

    required_cols = {"strategy", "precision", "recall", "f1", "auc_roc", "auc_pr", "threshold"}
    missing = required_cols - set(df_results.columns)
    assert not missing, f"FAIL — missing columns: {missing}"

    # Strategy 3 (SMOTE) should have more training rows than original
    smote_result = strategy_smote(X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    assert len(smote_result["y_train_resampled"]) > len(y_train_arr), (
        "FAIL — SMOTE did not increase training set size"
    )

    # Strategy 4 threshold must be in (0, 1)
    thresh4 = df_results.loc[df_results["strategy"] == "threshold_tuning", "threshold"].iloc[0]
    assert 0.0 < thresh4 < 1.0, f"FAIL — strategy 4 threshold {thresh4} not in (0, 1)"

    # SMOTE must not touch val (y_val_arr unchanged)
    assert np.array_equal(y_val_arr, y_val.to_numpy() if hasattr(y_val, 'to_numpy') else y_val), "FAIL — val set was modified"

    print("\nAll assertions passed.")

    # Print threshold details for strategy 4
    thr4_result = strategy_threshold_tuning(X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    print(f"Strategy 4 — business threshold: {thr4_result['threshold_business']:.3f}")
    print(f"Strategy 4 — F1 threshold:       {thr4_result['threshold_f1']:.3f}")

    print("\nimbalance.handler smoke test: OK")
