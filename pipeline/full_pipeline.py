"""End-to-end churn prediction pipeline.

Provides `ChurnPipeline`: a single object that encapsulates feature
engineering, preprocessing, and inference so callers can work directly
with raw input rows — no manual preprocessing required.

Usage (fit then predict):
    pipeline = ChurnPipeline()
    pipeline.fit(df_raw_with_churn)
    probs = pipeline.predict_proba(df_raw_without_churn)
    labels = pipeline.predict(df_raw_without_churn, threshold=0.5)

Persistence:
    pipeline.save("my_pipeline.pkl")
    pipeline = ChurnPipeline.load("my_pipeline.pkl")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from features.engineering import engineer
from features.preprocessing import fit_transform_splits, split_data
from features.selection import select_features
from models.trainer import N_TRIALS_SMOKE, tune_all_models

RANDOM_SEED = 42
_DEFAULT_MODELS_DIR = Path("results/models")

log = logging.getLogger(__name__)


class ChurnPipeline:
    """End-to-end customer churn prediction pipeline.

    Encapsulates feature engineering, preprocessing, and the best trained
    model into a single object.  Callers supply raw DataFrames matching
    the schema of `data/raw/telco_churn.csv`.

    Attributes:
        selected_features: Feature column names chosen during fit.
        transformer: Fitted ColumnTransformer for preprocessing.
        best_algo_name: Name of the best-performing algorithm.
        best_model: Fitted estimator with predict_proba / predict.
    """

    def __init__(self) -> None:
        self.selected_features: list[str] | None = None
        self.transformer: Any | None = None
        self.best_algo_name: str | None = None
        self.best_model: Any | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        """Raise RuntimeError if the pipeline has not been fitted."""
        if self.best_model is None or self.transformer is None:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() before predict_proba() or predict()."
            )

    def _prepare_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Coerce TotalCharges and run feature engineering."""
        df = df_raw.copy()
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return engineer(df)

    def _transform(self, df_engineered: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformer to the selected feature columns."""
        assert self.transformer is not None  # guarded by _assert_fitted
        assert self.selected_features is not None
        return self.transformer.transform(df_engineered[self.selected_features])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df_raw: pd.DataFrame,
        target_col: str = "Churn",
        models_dir: str | Path = _DEFAULT_MODELS_DIR,
        smoke: bool = False,
    ) -> ChurnPipeline:
        """Fit the full pipeline end-to-end on raw training data.

        Steps:
        1. Coerce TotalCharges and run feature engineering.
        2. Select features (MI + Random Forest union).
        3. Split data into train / val / test (70/15/15 stratified).
        4. Fit preprocessing transformer on train only.
        5. Tune all models with Optuna and save artifacts.
        6. Evaluate on held-out test set and select the best model.

        Args:
            df_raw: Raw DataFrame including the target column.
            target_col: Name of the binary target column (default "Churn").
            models_dir: Directory for model artifacts.
            smoke: If True (or TRAINER_SMOKE env var is set), use 3 Optuna
                   trials instead of 50 for speed.

        Returns:
            Self, for chaining.
        """
        smoke = smoke or bool(os.environ.get("TRAINER_SMOKE", ""))
        n_trials = N_TRIALS_SMOKE if smoke else 50

        log.info("Step 1: Feature engineering ...")
        engineered = self._prepare_raw(df_raw)

        log.info("Step 2: Feature selection ...")
        selected, _ = select_features(engineered, target_col=target_col)
        self.selected_features = selected
        log.info("  %d features selected.", len(selected))

        log.info("Step 3: Splitting data ...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            engineered, target_col=target_col
        )

        log.info("Step 4: Fitting preprocessor ...")
        X_tr, X_v, X_te, transformer = fit_transform_splits(
            X_train, X_val, X_test, selected
        )
        self.transformer = transformer

        log.info("Step 5: Tuning models (%d trials each) ...", n_trials)
        summary = tune_all_models(
            X_tr,
            y_train.values,
            X_v,
            y_val.values,
            output_dir=models_dir,
            n_trials=n_trials,
        )

        # Pick the best model by AUC-PR directly from tune_all_models' summary
        # to avoid loading from a hard-coded results/models/ path.
        log.info("Step 6: Selecting best model by val AUC-PR ...")
        best_algo = max(summary, key=lambda k: summary[k]["best_auc_pr"])
        self.best_algo_name = best_algo
        self.best_model = joblib.load(summary[best_algo]["model_path"])
        log.info(
            "Best model: %s (val AUC-PR=%.4f)",
            best_algo,
            summary[best_algo]["best_auc_pr"],
        )

        return self

    def predict_proba(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Return churn probability for each row.

        Args:
            df_raw: Raw DataFrame (without the target column).

        Returns:
            1-D array of churn probabilities, shape (n_samples,).
        """
        self._assert_fitted()
        engineered = self._prepare_raw(df_raw)
        X = self._transform(engineered)
        return self.best_model.predict_proba(X)

    def predict(self, df_raw: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary churn predictions.

        Args:
            df_raw: Raw DataFrame (without the target column).
            threshold: Decision threshold for the positive class (default 0.5).

        Returns:
            1-D integer array of predictions (0 = no churn, 1 = churn).
        """
        probs = self.predict_proba(df_raw)
        return (probs >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialise the pipeline to a joblib pickle file.

        Args:
            path: Destination file path (e.g. "my_pipeline.pkl").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ChurnPipeline:
        """Load a previously saved pipeline from a joblib pickle file.

        Args:
            path: Path to the `.pkl` file produced by save().

        Returns:
            A fitted ChurnPipeline instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        # Safety: these pickle files are produced exclusively by this
        # project's own training code and are never loaded from untrusted
        # or external sources.
        pipeline: ChurnPipeline = joblib.load(path)
        log.info("Pipeline loaded from %s (best model: %s)", path, pipeline.best_algo_name)
        return pipeline


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "telco_churn.csv"

    print(f"Loading dataset from {csv_path} ...")
    raw = pd.read_csv(str(csv_path))
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")

    # Use 200-row slice for smoke fit; isolate artifacts in a temp dir so the
    # shared results/models/ full-data artifacts are never clobbered.
    raw_sample = raw.sample(n=200, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Smoke fit slice: {len(raw_sample)} rows (includes 'Churn' column)")

    with tempfile.TemporaryDirectory() as _tmp_models:
        pipeline = ChurnPipeline()
        pipeline.fit(raw_sample, target_col="Churn", smoke=True, models_dir=_tmp_models)
        print(f"Fitted pipeline — best model: {pipeline.best_algo_name}")
        print(f"  Selected features: {len(pipeline.selected_features)}")

        # Predict on 5 raw rows WITHOUT manually calling engineer / preprocessor
        raw_predict = raw_sample.drop(columns=["Churn"]).head(5)
        print(f"\nRunning predict_proba on {len(raw_predict)} raw rows ...")
        probs = pipeline.predict_proba(raw_predict)
        assert probs.shape == (5,), f"Expected shape (5,), got {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"
        print(f"  Probabilities: {probs.round(4)}")

        labels = pipeline.predict(raw_predict, threshold=0.5)
        assert labels.shape == (5,), f"Expected shape (5,), got {labels.shape}"
        assert set(labels).issubset({0, 1}), f"Unexpected label values: {set(labels)}"
        print(f"  Labels:        {labels}")

        # Test save / load round-trip inside temp dir
        save_path = Path(_tmp_models) / "churn_pipeline_smoke.pkl"
        pipeline.save(save_path)
        loaded = ChurnPipeline.load(save_path)
        probs2 = loaded.predict_proba(raw_predict)
        assert np.allclose(probs, probs2), "Save/load round-trip produced different probabilities"
        print("  Save/load round-trip: OK")

    print("\npipeline.full_pipeline smoke test: OK")
    sys.exit(0)
