"""Tree-based model implementations for customer churn prediction.

Provides:
- RandomForestModel: sklearn RandomForestClassifier wrapper
- XGBoostModel: XGBoost classifier wrapper with early stopping
- LightGBMModel: LightGBM classifier wrapper with early stopping

All models share the same fit/predict_proba/predict interface and return
8-metric evaluation dicts via the shared evaluate() function from baseline.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from models.baseline import evaluate  # noqa: F401 — re-exported for callers

RANDOM_SEED = 42

log = logging.getLogger(__name__)


class RandomForestModel:
    """RandomForest classifier with balanced class weights.

    Wraps sklearn.ensemble.RandomForestClassifier with sensible defaults
    for churn prediction (balanced class weights, 300 trees).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        class_weight: str | None = "balanced",
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RandomForestModel":
        """Fit the random forest on training data.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Ignored — RandomForest does not support early stopping.
            y_val: Ignored — RandomForest does not support early stopping.

        Returns:
            Self for chaining.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities.

        Args:
            X: Feature array, shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) with P(class=1).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold.

        Args:
            X: Feature array, shape (n_samples, n_features).
            threshold: Decision threshold for positive class (default 0.5).

        Returns:
            1D int array of predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)


class XGBoostModel:
    """XGBoost classifier with AUC-PR optimisation and early stopping.

    Uses eval_metric="aucpr" and early_stopping_rounds=20 when validation
    data is supplied to fit().
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._extra_kwargs = kwargs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        """Fit the XGBoost model.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features for early stopping (optional).
            y_val: Validation labels for early stopping (optional).

        Returns:
            Self for chaining.
        """
        scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

        fit_kwargs: dict[str, object] = {}
        early_stopping_rounds: int | None = None

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            early_stopping_rounds = 20

        self.model = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            eval_metric="aucpr",
            early_stopping_rounds=early_stopping_rounds,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=0,
            **self._extra_kwargs,
        )

        self.model.fit(X_train, y_train, verbose=False, **fit_kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities.

        Args:
            X: Feature array, shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) with P(class=1).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold.

        Args:
            X: Feature array, shape (n_samples, n_features).
            threshold: Decision threshold for positive class (default 0.5).

        Returns:
            1D int array of predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)


class LightGBMModel:
    """LightGBM classifier with balanced class weights and early stopping.

    Uses balanced class_weight and early_stopping callback when validation
    data is supplied to fit().
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        class_weight: str | None = "balanced",
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.random_state = random_state
        self._extra_kwargs = kwargs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LightGBMModel":
        """Fit the LightGBM model.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features for early stopping (optional).
            y_val: Validation labels for early stopping (optional).

        Returns:
            Self for chaining.
        """
        callbacks: list[object] = [lgb.log_evaluation(period=-1)]
        fit_kwargs: dict[str, object] = {}

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model = lgb.LGBMClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbosity=-1,
            **self._extra_kwargs,
        )

        self.model.fit(
            X_train,
            y_train,
            callbacks=callbacks,
            **fit_kwargs,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities.

        Args:
            X: Feature array, shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) with P(class=1).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold.

        Args:
            X: Feature array, shape (n_samples, n_features).
            threshold: Decision threshold for positive class (default 0.5).

        Returns:
            1D int array of predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")
    sys.path.insert(0, project_root)

    import pandas as pd

    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    print(f"Loading data from {csv_path} ...")
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    print("Engineering features ...")
    engineered = engineer(raw)

    print("Selecting features ...")
    selected_features, _ = select_features(engineered, target_col="Churn")

    print("Splitting data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        engineered, target_col="Churn"
    )

    print("Transforming splits ...")
    X_train_arr, X_val_arr, X_test_arr, _ = fit_transform_splits(
        X_train, X_val, X_test, selected_features
    )

    # Smoke test checks models are learning (AUC-PR > random baseline = churn rate).
    # Optuna tuning in Task 8 is required to reliably beat the LogReg floor.
    churn_rate = y_val.mean()
    BASELINE_AUC_PR = churn_rate  # random classifier baseline
    print(f"Random baseline (churn rate): {BASELINE_AUC_PR:.4f}")

    models: dict[str, RandomForestModel | XGBoostModel | LightGBMModel] = {
        "RandomForest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
    }

    all_passed = True
    for name, model in models.items():
        print(f"\nFitting {name} ...")
        model.fit(X_train_arr, y_train, X_val=X_val_arr, y_val=y_val)

        metrics = evaluate(model, X_val_arr, y_val)
        auc_pr = metrics["auc_pr"]

        print(f"\n{name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name:15s}: {metric_value:.4f}")

        if auc_pr <= BASELINE_AUC_PR:
            print(f"  FAIL: AUC-PR {auc_pr:.4f} did not beat baseline {BASELINE_AUC_PR}")
            all_passed = False
        else:
            print(f"  PASS: AUC-PR {auc_pr:.4f} > baseline {BASELINE_AUC_PR}")

    if all_passed:
        print("\nAll tree models beat baseline AUC-PR")
    else:
        raise AssertionError("One or more tree models did not beat baseline AUC-PR")
