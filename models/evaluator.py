"""Model evaluation and visualisation for customer churn prediction.

Provides:
- evaluate_all_models: Load saved models, compute held-out test metrics.
- find_best_model: Identify the top performer by AUC-PR.
- plot_roc_curves: Overlaid ROC curves for all models.
- plot_pr_curves: Overlaid PR curves for all models.
- plot_confusion_matrix: Confusion matrix for the best model.
- plot_calibration_curves: Calibration curves for all models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve

from models.baseline import evaluate

RANDOM_SEED = 42
MODELS_DIR = Path("results/models")
RESULTS_DIR = Path("results")

_ALGO_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(algo_name: str) -> Any:
    """Load a saved model from results/models/{algo_name}_best.pkl.

    Args:
        algo_name: Short algorithm name (e.g. 'LogisticRegression').

    Returns:
        Fitted model object with predict_proba() and predict() interfaces.

    Raises:
        FileNotFoundError: If the model pickle does not exist.
    """
    path = MODELS_DIR / f"{algo_name}_best.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    # Safety: these pickle files are produced exclusively by the project's own
    # training pipeline (models/tuner.py) and are never loaded from untrusted
    # or external sources.  Loading them with joblib is intentional.
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_all_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Load each saved model and compute held-out test metrics.

    Iterates over the four trained algorithms, loads the best checkpoint,
    runs evaluate() from models.baseline, and collects all 8 metrics.

    Args:
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test target array of shape (n_samples,).

    Returns:
        Nested dict ``{algo_name: metrics_dict}`` where each metrics_dict
        contains the 8 keys returned by evaluate(): auc_roc, auc_pr, f1,
        precision, recall, accuracy, log_loss, brier_score.
    """
    results: dict[str, dict[str, float]] = {}
    for algo_name in _ALGO_NAMES:
        model = _load_model(algo_name)
        metrics = evaluate(model, X_test, y_test)
        results[algo_name] = metrics
        log.info(
            "%s — AUC-PR: %.4f  AUC-ROC: %.4f",
            algo_name,
            metrics["auc_pr"],
            metrics["auc_roc"],
        )
    return results


def find_best_model(
    metrics_dict: dict[str, dict[str, float]],
) -> tuple[str, Any]:
    """Return the algorithm name and loaded model with the highest AUC-PR.

    Args:
        metrics_dict: Output of evaluate_all_models().

    Returns:
        Tuple of (algo_name, loaded_model) for the best-performing model
        by AUC-PR on the held-out test set.

    Raises:
        ValueError: If metrics_dict is empty.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict is empty; cannot determine best model.")

    best_name = max(metrics_dict, key=lambda k: metrics_dict[k]["auc_pr"])
    best_model = _load_model(best_name)
    log.info(
        "Best model: %s (AUC-PR: %.4f)",
        best_name,
        metrics_dict[best_name]["auc_pr"],
    )
    return best_name, best_model


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _ensure_results_dir() -> None:
    """Create results/ directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics_dict: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> Path:
    """Plot overlaid ROC curves for all models with AUC-ROC in the legend.

    Args:
        X_test: Test feature array.
        y_test: Test target array.
        metrics_dict: Output of evaluate_all_models().
        save_path: Override output path (default: results/roc_curves.png).

    Returns:
        Path where the figure was saved.
    """
    _ensure_results_dir()
    out = save_path or (RESULTS_DIR / "roc_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")

    for algo_name in _ALGO_NAMES:
        if algo_name not in metrics_dict:
            continue
        model = _load_model(algo_name)
        y_prob = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = metrics_dict[algo_name]["auc_roc"]
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{algo_name} (AUC={auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Held-out Test Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curves → %s", out)
    return out


def plot_pr_curves(
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics_dict: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> Path:
    """Plot overlaid PR curves for all models with AUC-PR in the legend.

    Args:
        X_test: Test feature array.
        y_test: Test target array.
        metrics_dict: Output of evaluate_all_models().
        save_path: Override output path (default: results/pr_curves.png).

    Returns:
        Path where the figure was saved.
    """
    _ensure_results_dir()
    out = save_path or (RESULTS_DIR / "pr_curves.png")

    positive_rate = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=positive_rate, color="k", linestyle="--", linewidth=0.8, label="Random baseline")

    for algo_name in _ALGO_NAMES:
        if algo_name not in metrics_dict:
            continue
        model = _load_model(algo_name)
        y_prob = model.predict_proba(X_test)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        auc_val = metrics_dict[algo_name]["auc_pr"]
        ax.plot(recall_vals, precision_vals, linewidth=1.5, label=f"{algo_name} (AUC={auc_val:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Held-out Test Set")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved PR curves → %s", out)
    return out


def plot_confusion_matrix(
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_name: str,
    best_model: Any,
    save_path: Path | None = None,
) -> Path:
    """Plot confusion matrix for the best model (by AUC-PR).

    Args:
        X_test: Test feature array.
        y_test: Test target array.
        best_name: Algorithm name of the best model.
        best_model: Fitted model with a predict() method.
        save_path: Override output path (default: results/confusion_matrix.png).

    Returns:
        Path where the figure was saved.
    """
    _ensure_results_dir()
    out = save_path or (RESULTS_DIR / "confusion_matrix.png")

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name} (Best by AUC-PR)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out)
    return out


def plot_calibration_curves(
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path | None = None,
    n_bins: int = 10,
) -> Path:
    """Plot calibration curves for all models.

    Args:
        X_test: Test feature array.
        y_test: Test target array.
        save_path: Override output path (default: results/calibration_curve.png).
        n_bins: Number of bins for calibration curve (default 10).

    Returns:
        Path where the figure was saved.
    """
    _ensure_results_dir()
    out = save_path or (RESULTS_DIR / "calibration_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")

    for algo_name in _ALGO_NAMES:
        model = _load_model(algo_name)
        y_prob = model.predict_proba(X_test)
        fraction_of_positives, mean_predicted = calibration_curve(
            y_test, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_predicted, fraction_of_positives, marker="o", linewidth=1.5, label=algo_name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — Held-out Test Set")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved calibration curves → %s", out)
    return out


# ---------------------------------------------------------------------------
# Convenience: run full evaluation pipeline
# ---------------------------------------------------------------------------


def run_full_evaluation(
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Run evaluate_all_models and generate all four plots.

    Args:
        X_test: Test feature array.
        y_test: Test target array.

    Returns:
        metrics_dict from evaluate_all_models().
    """
    metrics_dict = evaluate_all_models(X_test, y_test)
    best_name, best_model = find_best_model(metrics_dict)

    plot_roc_curves(X_test, y_test, metrics_dict)
    plot_pr_curves(X_test, y_test, metrics_dict)
    plot_confusion_matrix(X_test, y_test, best_name, best_model)
    plot_calibration_curves(X_test, y_test)

    return metrics_dict


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)

    import pandas as pd

    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    csv_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")
    print(f"Loading dataset from {csv_path} ...")
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    # Use the same 500-row sample as the trainer smoke run so that the saved
    # model artifacts (43 features) match the preprocessing output here.
    raw = raw.sample(n=500, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Smoke slice: {len(raw)} rows")

    print("Engineering features ...")
    engineered = engineer(raw)

    print("Selecting features ...")
    selected_features, _ = select_features(engineered, target_col="Churn")

    print("Splitting data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered, target_col="Churn")

    print("Preprocessing ...")
    _, _, X_test_arr, _ = fit_transform_splits(X_train, X_val, X_test, selected_features)

    print("Running evaluate_all_models ...")
    metrics_dict = evaluate_all_models(X_test_arr, y_test.values)

    _EXPECTED_KEYS = {
        "auc_roc", "auc_pr", "f1", "precision", "recall",
        "accuracy", "log_loss", "brier_score",
    }
    for algo_name, m in metrics_dict.items():
        missing = _EXPECTED_KEYS - set(m.keys())
        assert not missing, f"FAIL — {algo_name} missing keys: {missing}"
        print(f"  {algo_name}: AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}")

    print("Finding best model ...")
    best_name, best_model = find_best_model(metrics_dict)
    print(f"  Best model: {best_name}")

    print("Generating plots ...")
    plot_roc_curves(X_test_arr, y_test.values, metrics_dict)
    plot_pr_curves(X_test_arr, y_test.values, metrics_dict)
    plot_confusion_matrix(X_test_arr, y_test.values, best_name, best_model)
    plot_calibration_curves(X_test_arr, y_test.values)

    _EXPECTED_FIGS = [
        RESULTS_DIR / "roc_curves.png",
        RESULTS_DIR / "pr_curves.png",
        RESULTS_DIR / "confusion_matrix.png",
        RESULTS_DIR / "calibration_curve.png",
    ]
    for fig_path in _EXPECTED_FIGS:
        assert fig_path.exists(), f"FAIL — figure not found: {fig_path}"
        print(f"  Verified figure exists: {fig_path}")

    print("\nmodels.evaluator smoke test: OK")
