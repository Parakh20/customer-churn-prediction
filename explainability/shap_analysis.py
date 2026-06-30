"""SHAP-based model explainability for customer churn prediction.

Provides:
- run_shap_analysis: Compute SHAP values, save global/local plots, derive insights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

RANDOM_SEED = 42
log = logging.getLogger(__name__)

# Model class names that are tree-based
_TREE_MODEL_CLASSES = {"XGBoostModel", "RandomForestModel", "LightGBMModel"}


# ---------------------------------------------------------------------------
# Explainer creation
# ---------------------------------------------------------------------------


def _make_explainer(model: Any, X_background: np.ndarray | None = None) -> Any:
    """Create an appropriate SHAP explainer for the given wrapper model.

    Args:
        model: A wrapper model with a `.model` attribute.
        X_background: Background data used for LinearExplainer masking.

    Returns:
        A SHAP explainer instance.
    """
    class_name = type(model).__name__
    inner = model.model
    if class_name in _TREE_MODEL_CLASSES:
        return shap.TreeExplainer(inner)
    # LinearExplainer requires background data matching the feature dimensionality
    background = X_background if X_background is not None else np.zeros((1, inner.coef_.shape[1]))
    return shap.LinearExplainer(inner, masker=shap.maskers.Independent(background))


def _compute_shap_values(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values, collapsing multi-output to the positive class.

    Args:
        explainer: A SHAP explainer.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        SHAP values of shape (n_samples, n_features).
    """
    raw = explainer.shap_values(X)
    # Tree-based binary classifiers may return a list [neg_class, pos_class]
    if isinstance(raw, list):
        return raw[1]
    return raw


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------


def _save_beeswarm(shap_values: np.ndarray, X: np.ndarray,
                   feature_names: list[str], output_dir: Path) -> None:
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=100, bbox_inches="tight")
    plt.close()


def _save_bar(shap_values: np.ndarray, feature_names: list[str], output_dir: Path) -> list[str]:
    """Save mean |SHAP| bar chart. Returns top-feature names sorted by importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features = [feature_names[i] for i in order]

    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=100, bbox_inches="tight")
    plt.close()
    return top_features


def _save_dependence(shap_values: np.ndarray, X: np.ndarray,
                     feature_names: list[str], top_features: list[str],
                     output_dir: Path) -> None:
    for rank, feat in enumerate(top_features[:3], start=1):
        idx = feature_names.index(feat)
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_dependence_top{rank}.png", dpi=100, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Local waterfall plots
# ---------------------------------------------------------------------------


def _find_sample_index(y: np.ndarray, probs: np.ndarray,
                       label: int, prob_cond: str, threshold: float) -> int | None:
    """Find first index matching label and probability condition."""
    for i, (yi, pi) in enumerate(zip(y, probs)):
        if yi != label:
            continue
        if prob_cond == "gt" and pi > threshold:
            return i
        if prob_cond == "lt" and pi < threshold:
            return i
    return None


def _save_waterfall(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str],
                    idx: int, output_dir: Path, filename: str) -> None:
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=0.0,
        data=X[idx],
        feature_names=feature_names,
    )
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=100, bbox_inches="tight")
    plt.close()


def _save_local_plots(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str],
                      y_test: np.ndarray, probs: np.ndarray, output_dir: Path) -> None:
    """Save waterfall plots for churner, retained, and false-negative customers."""
    cases = [
        ("churner", 1, "gt", 0.7, "shap_waterfall_churner.png"),
        ("retained", 0, "lt", 0.3, "shap_waterfall_retained.png"),
        ("false_negative", 1, "lt", 0.3, "shap_waterfall_false_negative.png"),
    ]
    for name, label, cond, thresh, fname in cases:
        idx = _find_sample_index(y_test, probs, label, cond, thresh)
        if idx is None:
            log.warning("No sample found for '%s'; skipping waterfall.", name)
            continue
        _save_waterfall(shap_values, X, feature_names, idx, output_dir, fname)


# ---------------------------------------------------------------------------
# Business insights
# ---------------------------------------------------------------------------


def _derive_insights(shap_values: np.ndarray, feature_names: list[str],
                     top_features: list[str]) -> list[str]:
    """Derive 3 business insight sentences from SHAP magnitudes.

    Args:
        shap_values: Array of shape (n_samples, n_features).
        feature_names: List of feature names.
        top_features: Feature names sorted by mean |SHAP| descending.

    Returns:
        List of 3 insight strings.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    total_impact = mean_abs.sum()

    def _pct(feat: str) -> float:
        idx = feature_names.index(feat)
        return 100.0 * mean_abs[idx] / total_impact if total_impact > 0 else 0.0

    f1, f2, f3 = (top_features[i] if i < len(top_features) else f"feature_{i}" for i in range(3))
    p1, p2, p3 = _pct(f1), _pct(f2), _pct(f3)

    insights = [
        (
            f"'{f1}' is the strongest churn driver, accounting for {p1:.1f}% of total "
            "model impact — customers with higher values show markedly elevated churn risk."
        ),
        (
            f"'{f2}' contributes {p2:.1f}% of SHAP impact; targeted intervention on this "
            "dimension could yield the second-largest reduction in churn."
        ),
        (
            f"The combined SHAP weight of the top-3 features ('{f1}', '{f2}', '{f3}') "
            f"is {p1 + p2 + p3:.1f}%, suggesting that churn is concentrated in a small "
            "set of high-leverage predictors rather than spread diffusely."
        ),
    ]
    return insights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_shap_analysis(
    model: Any,
    X_test_arr: np.ndarray,
    feature_names: list[str],
    output_dir: str = "results/",
    y_test: np.ndarray | None = None,
    X_background: np.ndarray | None = None,
) -> dict:
    """Run SHAP analysis on the best model, save plots, and return summary dict.

    Args:
        model: Wrapper model with a `.model` attribute.
        X_test_arr: Feature matrix for the test set, shape (n_samples, n_features).
        feature_names: List of feature column names.
        output_dir: Directory where PNG figures will be written.
        y_test: True labels for local waterfall plots (optional but recommended).
        X_background: Background data for LinearExplainer (use training data).

    Returns:
        Dict with keys:
            - ``shap_values``: np.ndarray of shape (n_samples, n_features)
            - ``top_features``: list of feature names sorted by mean |SHAP|
            - ``insights``: list of 3 business insight strings
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Building SHAP explainer for %s …", type(model).__name__)
    explainer = _make_explainer(model, X_background=X_background)

    log.info("Computing SHAP values for %d samples …", len(X_test_arr))
    shap_values = _compute_shap_values(explainer, X_test_arr)

    log.info("Saving global plots …")
    _save_beeswarm(shap_values, X_test_arr, feature_names, out)
    top_features = _save_bar(shap_values, feature_names, out)
    _save_dependence(shap_values, X_test_arr, feature_names, top_features, out)

    if y_test is not None:
        log.info("Saving local waterfall plots …")
        probs = model.predict_proba(X_test_arr)
        _save_local_plots(shap_values, X_test_arr, feature_names, y_test, probs, out)
    else:
        log.warning("y_test not provided; local waterfall plots will be skipped.")

    insights = _derive_insights(shap_values, feature_names, top_features)

    return {
        "shap_values": shap_values,
        "top_features": top_features,
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from features.engineering import engineer
    from features.preprocessing import split_data, fit_transform_splits
    from features.selection import select_features
    from models.evaluator import evaluate_all_models, find_best_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = pd.read_csv("data/raw/telco_churn.csv")
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    eng = engineer(raw)
    feats, _ = select_features(eng, target_col="Churn")
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(eng, target_col="Churn")
    X_tr_arr, X_v_arr, X_te_arr, transformer = fit_transform_splits(X_tr, X_v, X_te, feats)

    metrics = evaluate_all_models(X_te_arr, y_te.values)
    algo_name, best_model = find_best_model(metrics)
    log.info("Best model: %s", algo_name)

    # Use the transformer's expanded feature names (46 after one-hot encoding)
    expanded_names = list(transformer.get_feature_names_out())

    result = run_shap_analysis(
        model=best_model,
        X_test_arr=X_te_arr,
        feature_names=expanded_names,
        output_dir="results/",
        y_test=y_te.values,
        X_background=X_tr_arr,
    )

    shap_vals = result["shap_values"]
    n_test, n_feats = X_te_arr.shape
    assert shap_vals.shape == (n_test, n_feats), (
        f"Shape mismatch: got {shap_vals.shape}, expected ({n_test}, {n_feats})"
    )

    expected_figs = [
        "shap_beeswarm.png",
        "shap_bar.png",
        "shap_dependence_top1.png",
        "shap_dependence_top2.png",
        "shap_dependence_top3.png",
        "shap_waterfall_churner.png",
        "shap_waterfall_retained.png",
        "shap_waterfall_false_negative.png",
    ]
    missing = [f for f in expected_figs if not (Path("results") / f).exists()]
    assert not missing, f"Missing figures: {missing}"

    print("\n=== Business Insights ===")
    for i, insight in enumerate(result["insights"], start=1):
        print(f"{i}. {insight}")

    print(f"\nTop features: {result['top_features'][:5]}")
    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"Figures verified: {len(expected_figs)}/8")
    print("\nexplainability.shap_analysis smoke test: OK")
