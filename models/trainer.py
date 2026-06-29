"""Model training orchestration with Optuna hyperparameter tuning.

Provides:
- tune_all_models: run Optuna Bayesian search for all 4 algorithms,
  save best model + params to results/models/, return summary dict.

Primary metric: AUC-PR (precision-recall AUC).
Business note: false negatives cost 5× false positives — AUC-PR is the
right optimisation target because it focuses on the positive (churn) class
without inflating scores via true negatives.

Usage:
    python -m models.trainer            # full run (50 trials/model)
    python -m models.trainer --smoke    # smoke run (3 trials/model)
    TRAINER_SMOKE=1 python -m models.trainer  # env-var smoke trigger
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from models.baseline import LogisticRegressionBaseline
from models.tree_models import LightGBMModel, RandomForestModel, XGBoostModel

RANDOM_SEED = 42
N_SPLITS = 5
N_TRIALS_FULL = 50
N_TRIALS_SMOKE = 3

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# AUC-PR helper
# ---------------------------------------------------------------------------


def _auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


# ---------------------------------------------------------------------------
# 5-fold stratified CV scorer
# ---------------------------------------------------------------------------


def _cross_val_auc_pr(
    model_cls: type,
    model_kwargs: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Run 5-fold stratified CV and return mean AUC-PR.

    Args:
        model_cls: Model class to instantiate (e.g. LogisticRegressionBaseline).
        model_kwargs: Constructor kwargs for the model.
        X: Combined train+val feature array.
        y: Combined train+val labels.

    Returns:
        Mean AUC-PR across 5 folds.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_fold_train, y_fold_train, X_val=X_fold_val, y_val=y_fold_val)
        y_score = model.predict_proba(X_fold_val)
        fold_scores.append(_auc_pr(y_fold_val, y_score))
        log.debug("  fold %d AUC-PR=%.4f", fold_idx + 1, fold_scores[-1])

    return float(np.mean(fold_scores))


def _logreg_cv(model_kwargs: dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
    """Cross-validate LogisticRegressionBaseline (no X_val/y_val in fit)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model = LogisticRegressionBaseline(**model_kwargs)
        model.fit(X_fold_train, y_fold_train)
        y_score = model.predict_proba(X_fold_val)
        fold_scores.append(_auc_pr(y_fold_val, y_score))

    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Per-algorithm objective factories
# ---------------------------------------------------------------------------


def _logreg_objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray
) -> float:
    """Optuna objective for LogisticRegressionBaseline.

    Search space:
        C             : [0.01, 10] log-uniform
        class_weight  : {None, 'balanced'}
    """
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    return _logreg_cv(
        {"C": C, "class_weight": class_weight, "random_state": RANDOM_SEED},
        X,
        y,
    )


def _rf_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for RandomForestModel.

    Search space:
        n_estimators      : [100, 400]
        max_depth         : [3, 20]
        min_samples_leaf  : [1, 20]
    """
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    return _cross_val_auc_pr(
        RandomForestModel,
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": RANDOM_SEED,
        },
        X,
        y,
    )


def _xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoostModel.

    Search space:
        learning_rate     : [0.01, 0.3] log-uniform
        n_estimators      : [100, 500]
        max_depth         : [3, 10]
        subsample         : [0.6, 1.0]
        colsample_bytree  : [0.6, 1.0]
    """
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

    return _cross_val_auc_pr(
        XGBoostModel,
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": RANDOM_SEED,
        },
        X,
        y,
    )


def _lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for LightGBMModel.

    Search space (same as XGB plus num_leaves):
        learning_rate     : [0.01, 0.3] log-uniform
        n_estimators      : [100, 500]
        max_depth         : [3, 10]
        subsample         : [0.6, 1.0]
        colsample_bytree  : [0.6, 1.0]
        num_leaves        : [20, 100]
    """
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    num_leaves = trial.suggest_int("num_leaves", 20, 100)

    return _cross_val_auc_pr(
        LightGBMModel,
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": RANDOM_SEED,
        },
        X,
        y,
    )


# ---------------------------------------------------------------------------
# Per-algorithm final model builders (fit on all train+val)
# ---------------------------------------------------------------------------


def _build_logreg(params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    model = LogisticRegressionBaseline(
        C=params["C"],
        class_weight=params["class_weight"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


def _build_rf(params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    model = RandomForestModel(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


def _build_xgb(params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    model = XGBoostModel(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


def _build_lgbm(params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    model = LightGBMModel(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_ALGO_REGISTRY: list[
    tuple[
        str,                                     # algorithm name
        Any,                                     # objective fn (trial, X, y) -> float
        Any,                                     # builder fn (params, X, y) -> model
    ]
] = [
    ("LogisticRegression", _logreg_objective, _build_logreg),
    ("RandomForest", _rf_objective, _build_rf),
    ("XGBoost", _xgb_objective, _build_xgb),
    ("LightGBM", _lgbm_objective, _build_lgbm),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str | Path = "results/models",
    n_trials: int = N_TRIALS_FULL,
) -> dict[str, dict[str, Any]]:
    """Tune all 4 algorithms via Optuna, save best model and params.

    CV is run on train+val combined (70% + 15% = 85% of full data) using
    5-fold stratified CV so the validation leakage is contained within
    the Optuna loop.

    After tuning, the best model for each algorithm is re-fitted on the
    entire train+val array and saved as a joblib pickle.

    Args:
        X_train: Training feature array, shape (n_train, n_features).
        y_train: Training labels, shape (n_train,).
        X_val:   Validation feature array, shape (n_val, n_features).
        y_val:   Validation labels, shape (n_val,).
        output_dir: Directory for model artefacts (created if absent).
        n_trials: Number of Optuna trials per algorithm (default 50).

    Returns:
        Summary dict keyed by algorithm name:
        {
            "LogisticRegression": {
                "best_auc_pr": float,
                "best_params": dict,
                "model_path": str,
            },
            ...
        }
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Merge train+val for cross-validation
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    summary: dict[str, dict[str, Any]] = {}

    for algo_name, objective_fn, build_fn in _ALGO_REGISTRY:
        log.info("Tuning %s (%d trials) ...", algo_name, n_trials)

        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(
            lambda trial, fn=objective_fn: fn(trial, X_all, y_all),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_auc_pr = study.best_value

        log.info(
            "%s best AUC-PR=%.4f params=%s", algo_name, best_auc_pr, best_params
        )

        # Re-fit on full train+val with best params
        final_model = build_fn(best_params, X_all, y_all)

        # Persist model
        model_path = out_path / f"{algo_name}_best.pkl"
        joblib.dump(final_model, model_path)

        # Persist params
        params_path = out_path / f"{algo_name}_best_params.json"
        with params_path.open("w") as fh:
            json.dump({"best_auc_pr": best_auc_pr, "best_params": best_params}, fh, indent=2)

        summary[algo_name] = {
            "best_auc_pr": best_auc_pr,
            "best_params": best_params,
            "model_path": str(model_path),
        }

    return summary


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train and tune churn models.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=bool(os.environ.get("TRAINER_SMOKE", "")),
        help="Smoke mode: 3 Optuna trials per model (default: 50).",
    )
    args = parser.parse_args()

    n_trials = N_TRIALS_SMOKE if args.smoke else N_TRIALS_FULL

    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "telco_churn.csv"

    sys.path.insert(0, str(project_root))
    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    log.info("Loading %s ...", csv_path)
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    # Smoke mode: use a tiny slice (500 rows) for speed
    if args.smoke:
        raw = raw.sample(n=500, random_state=RANDOM_SEED).reset_index(drop=True)
        log.info("Smoke mode: using %d rows", len(raw))

    log.info("Engineering features ...")
    engineered = engineer(raw)

    log.info("Selecting features ...")
    selected_features, _ = select_features(engineered, target_col="Churn")
    log.info("  %d features selected", len(selected_features))

    log.info("Splitting data ...")
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(
        engineered, target_col="Churn"
    )

    log.info("Transforming splits ...")
    X_train_arr, X_val_arr, _X_test_arr, _ = fit_transform_splits(
        X_train_df, X_val_df, X_test_df, selected_features
    )

    log.info(
        "Shapes — train: %s  val: %s", X_train_arr.shape, X_val_arr.shape
    )

    out_dir = project_root / "results" / "models"
    log.info("Output directory: %s", out_dir)

    results = tune_all_models(
        X_train_arr,
        y_train,
        X_val_arr,
        y_val,
        output_dir=out_dir,
        n_trials=n_trials,
    )

    print("\n=== Tuning Summary ===")
    for algo, info in results.items():
        print(
            f"  {algo:22s}  AUC-PR={info['best_auc_pr']:.4f}"
            f"  model={info['model_path']}"
        )

    # Verify artefacts exist and AUC-PR is sane
    for algo, info in results.items():
        assert Path(info["model_path"]).exists(), f"Missing model: {info['model_path']}"
        params_path = Path(info["model_path"]).with_suffix("").parent / f"{algo}_best_params.json"
        assert params_path.exists(), f"Missing params: {params_path}"
        assert info["best_auc_pr"] > 0.0, f"AUC-PR should be > 0 for {algo}"

    print("\nmodels.trainer smoke test: OK")
