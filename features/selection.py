"""Feature selection module for customer churn prediction.

Three selection methods are applied:
1. Mutual information (MI) — top-20 features by MI score vs target.
2. Correlation matrix — flags redundant pairs with |r| > 0.9; for each
   redundant pair the member with *lower* MI score is noted as removable.
3. Random Forest importances — top-20 features by mean decrease in impurity.

The public API returns the *union* of the MI top-20 and RF top-20, plus the
list of redundant pairs for reporting purposes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

RANDOM_SEED = 42
_TOP_N = 20
_CORR_THRESHOLD = 0.9
_ID_COLS = {"customerID"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_feature_pool(df: pd.DataFrame, target_col: str) -> list[str]:
    """Return feature column names (exclude target and ID columns)."""
    exclude = _ID_COLS | {target_col}
    return [c for c in df.columns if c not in exclude]


def _binarise_target(series: pd.Series) -> pd.Series:
    """Map Yes/No strings to 1/0; leave numeric series unchanged."""
    if series.dtype == object:
        return series.map({"Yes": 1, "No": 0}).astype(int)
    return series.astype(int)


def _encode_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a fully numeric DataFrame for the given feature columns.

    Categorical columns are encoded with OrdinalEncoder.
    NaN values are filled with the column median after encoding.
    """
    X = df[feature_cols].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))

    # Fill remaining NaN with column median
    X = X.apply(lambda col: col.fillna(col.median()))

    return X.astype(float)


def _top_n_by_score(scores: pd.Series, n: int) -> list[str]:
    """Return the names of the top-n features sorted by descending score."""
    return scores.nlargest(n).index.tolist()


# ---------------------------------------------------------------------------
# Method 1 — Mutual information
# ---------------------------------------------------------------------------


def _mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Compute MI scores for all features; return a Series indexed by col name."""
    mi_values = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    return pd.Series(mi_values, index=X.columns)


# ---------------------------------------------------------------------------
# Method 2 — Correlation matrix
# ---------------------------------------------------------------------------


def _find_redundant_pairs(
    X: pd.DataFrame,
    mi_scores: pd.Series,
    threshold: float = _CORR_THRESHOLD,
) -> list[tuple[str, str]]:
    """Return pairs (lower_mi_feat, higher_mi_feat) where |r| > threshold.

    The first element of each pair is the feature with *lower* MI score
    (candidate for removal); the second has higher MI (candidate to keep).
    """
    corr = X.corr().abs()
    pairs: list[tuple[str, str]] = []
    cols = corr.columns.tolist()

    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            if corr.loc[col_a, col_b] > threshold:
                # Keep the feature with higher MI score
                if mi_scores.get(col_a, 0) >= mi_scores.get(col_b, 0):
                    pairs.append((col_b, col_a))  # (to_drop_candidate, to_keep)
                else:
                    pairs.append((col_a, col_b))
    return pairs


# ---------------------------------------------------------------------------
# Method 3 — Random Forest importances
# ---------------------------------------------------------------------------


def _random_forest_importances(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Fit a RandomForestClassifier and return feature importances."""
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Select features for churn prediction using MI and Random Forest.

    Steps:
    1. Build the feature pool (all columns except target and ID columns).
    2. Binarise target (Yes → 1, No → 0).
    3. Encode categoricals and fill NaN with median.
    4. Compute mutual information scores → top-20.
    5. Flag correlation-redundant pairs (|r| > 0.9).
    6. Fit Random Forest → top-20 by importance.
    7. Return union of both top-20 lists (deduped, sorted) plus redundant pairs.

    Args:
        df: Engineered DataFrame (original + derived features).
        target_col: Name of the binary target column (default "Churn").

    Returns:
        A tuple of:
        - selected: sorted list of selected feature names (union of MI & RF top-20).
        - redundant_pairs: list of (lower_mi_feat, higher_mi_feat) tuples where
          |r| > 0.9.
    """
    feature_cols = _build_feature_pool(df, target_col)
    if not feature_cols:
        raise ValueError("No feature columns found after excluding target and ID cols.")

    y = _binarise_target(df[target_col])
    X = _encode_features(df, feature_cols)

    log.info("Computing mutual information scores for %d features ...", len(feature_cols))
    mi_scores = _mutual_information(X, y)
    mi_top20 = _top_n_by_score(mi_scores, _TOP_N)

    log.info("Detecting redundant feature pairs (|r| > %.1f) ...", _CORR_THRESHOLD)
    redundant_pairs = _find_redundant_pairs(X, mi_scores)
    if redundant_pairs:
        log.info("Redundant pairs found:")
        for pair in redundant_pairs:
            log.info("  |r| > %.1f  →  drop candidate: %s  |  keep: %s", _CORR_THRESHOLD, *pair)

    log.info("Fitting RandomForest on %d features ...", len(feature_cols))
    rf_scores = _random_forest_importances(X, y)
    rf_top20 = _top_n_by_score(rf_scores, _TOP_N)

    selected = sorted(set(mi_top20) | set(rf_top20))

    log.info(
        "Selected %d features (union of MI top-%d and RF top-%d).",
        len(selected),
        _TOP_N,
        _TOP_N,
    )
    return selected, redundant_pairs


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

    # Import engineer from sibling module
    sys.path.insert(0, project_root)
    from features.engineering import engineer  # noqa: E402

    print("Running engineer() ...")
    engineered = engineer(raw)

    print("Running select_features() ...")
    selected, redundant_pairs = select_features(engineered, target_col="Churn")

    # --- assertions ---
    assert len(selected) > 0, "FAIL — selected feature list is empty"
    assert len(selected) >= 10, f"FAIL — expected ≥10 features, got {len(selected)}"
    assert "customerID" not in selected, "FAIL — customerID leaked into features"
    assert "Churn" not in selected, "FAIL — target column leaked into features"
    deduped_check = sorted(set(selected))
    assert selected == deduped_check, "FAIL — selected list is not deduped/sorted"

    print("\n=== Selected features ===")
    for i, feat in enumerate(selected, 1):
        print(f"  {i:2d}. {feat}")

    print(f"\nTotal selected: {len(selected)}")

    print("\n=== Redundant pairs (|r| > 0.9) ===")
    if redundant_pairs:
        for drop_cand, keep in redundant_pairs:
            print(f"  drop candidate: {drop_cand!r}  |  keep: {keep!r}")
    else:
        print("  (none)")

    print("\nfeatures.selection smoke test: OK")
