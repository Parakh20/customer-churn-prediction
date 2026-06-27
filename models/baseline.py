"""Logistic Regression baseline model for customer churn prediction.

Provides:
- LogisticRegressionBaseline: Wrapper around sklearn's LogisticRegression
- evaluate: Standard evaluation function returning 8 key metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)

RANDOM_SEED = 42


class LogisticRegressionBaseline:
    """Logistic Regression baseline model with interpretability support.

    Wraps sklearn.linear_model.LogisticRegression with configurable
    class_weight, regularization (C), and hyperparameters.
    """

    def __init__(
        self,
        class_weight: str | None = None,
        C: float = 1.0,
        random_state: int = RANDOM_SEED,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the baseline model.

        Args:
            class_weight: One of {None, 'balanced'} to handle class imbalance.
            C: Inverse regularization strength (default 1.0).
            random_state: Random seed for reproducibility.
            max_iter: Maximum iterations for solver convergence.
        """
        self.class_weight = class_weight
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter

        self.model = LogisticRegression(
            class_weight=class_weight,
            C=C,
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self._feature_names: list[str] | None = None

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> LogisticRegressionBaseline:
        """Fit the logistic regression model on training data.

        Args:
            X_train: Training feature array of shape (n_samples, n_features).
            y_train: Training target array of shape (n_samples,).

        Returns:
            Self for chaining.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class for each sample.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) with probability of class=1.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold.

        Args:
            X: Feature array of shape (n_samples, n_features).
            threshold: Decision threshold for positive class (default 0.5).

        Returns:
            1D array of binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_coefficients(self) -> pd.DataFrame:
        """Return DataFrame of feature coefficients sorted by absolute value.

        Returns:
            DataFrame with columns ['feature_index', 'coefficient'],
            sorted by |coefficient| in descending order.
        """
        coefs = self.model.coef_[0]
        abs_coefs = np.abs(coefs)
        sorted_idx = np.argsort(-abs_coefs)

        return pd.DataFrame(
            {
                "feature_index": sorted_idx,
                "coefficient": coefs[sorted_idx],
            }
        )


def evaluate(
    model: LogisticRegressionBaseline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate model on validation data returning 8 standard metrics.

    Args:
        model: Fitted LogisticRegressionBaseline instance.
        X_val: Validation feature array of shape (n_samples, n_features).
        y_val: Validation target array of shape (n_samples,).
        threshold: Decision threshold for binary predictions (default 0.5).

    Returns:
        Dictionary with keys:
        - auc_roc: Area under receiver operating characteristic curve
        - auc_pr: Area under precision-recall curve
        - f1: F1 score
        - precision: Precision score
        - recall: Recall score
        - accuracy: Accuracy score
        - log_loss: Log loss (cross-entropy)
        - brier_score: Brier score (mean squared error of probabilities)
    """
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val, threshold=threshold)

    # AUC-ROC
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc_roc = auc(fpr, tpr)

    # AUC-PR
    from sklearn.metrics import precision_recall_curve

    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
    auc_pr = auc(recall_vals, precision_vals)

    # F1, Precision, Recall, Accuracy
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall_val = recall_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)

    # Log Loss and Brier Score
    ll = log_loss(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall_val,
        "accuracy": accuracy,
        "log_loss": ll,
        "brier_score": brier,
    }


if __name__ == "__main__":
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")

    print(f"Loading dataset from {csv_path} ...")
    raw = pd.read_csv(csv_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    sys.path.insert(0, project_root)
    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    print("Running engineer() ...")
    engineered = engineer(raw)

    print("Running select_features() ...")
    selected_features, _ = select_features(engineered, target_col="Churn")
    print(f"  Selected {len(selected_features)} features")

    print("Splitting data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        engineered, target_col="Churn"
    )

    print("Fitting preprocessor and transforming splits ...")
    X_train_arr, X_val_arr, X_test_arr, _ = fit_transform_splits(
        X_train, X_val, X_test, selected_features
    )

    print(f"  X_train shape: {X_train_arr.shape}")
    print(f"  X_val   shape: {X_val_arr.shape}")

    print("Fitting baseline model ...")
    baseline = LogisticRegressionBaseline(class_weight=None)
    baseline.fit(X_train_arr, y_train)

    print("Evaluating on validation set ...")
    metrics = evaluate(baseline, X_val_arr, y_val)

    print("\nValidation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:15s}: {metric_value:.4f}")

    # Assert AUC-ROC > 0.5
    assert metrics["auc_roc"] > 0.5, f"FAIL — AUC-ROC {metrics['auc_roc']:.4f} not > 0.5"
    print("\nAUC-ROC > 0.5 assertion passed.")

    print("\nmodels.baseline smoke test: OK")
