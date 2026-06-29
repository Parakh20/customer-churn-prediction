# Customer Churn Prediction — Implementation Plan

End-to-end churn pipeline: rigorous feature engineering, class-imbalance
handling, model comparison (LogReg / RF / XGBoost / LightGBM), SHAP
explainability, Streamlit dashboard. Built via
`superpowers:subagent-driven-development` — see `../SUBAGENT_DEV_PLAN.md` for
workflow, model budget, and the per-project pinned-venv rule.

## Global constraints (hand to every reviewer)

- Python 3.12, isolated `.venv`, **pinned** `requirements.txt`.
- Primary optimization metric: **AUC-PR** (imbalanced data). Report AUC-ROC,
  AUC-PR, F1, precision, recall, accuracy, log loss, Brier score.
- Cost asymmetry: a false negative costs **5×** a false positive (drives
  threshold tuning and the "business metric").
- Splits: **70/15/15 train/val/test, stratified** on churn. `RANDOM_SEED=42`.
- Every module has a `__main__` smoke test on a tiny slice.
- Datasets fetched by `data/download.py`, never committed.

## Target structure

```
customer_churn_prediction/
├── data/{raw,processed}/  data/download.py
├── features/{engineering,selection,preprocessing}.py  __init__.py
├── imbalance/handler.py  __init__.py
├── models/{baseline,tree_models,trainer,evaluator}.py  __init__.py
├── explainability/shap_analysis.py  __init__.py
├── pipeline/full_pipeline.py  __init__.py
├── visualisation/plots.py
├── dashboard/app.py
├── results/  requirements.txt  README.md  .gitignore
```

---

## Tasks

### Task 0 — Env + scaffold + requirements  ·  model: haiku
Create `.venv`, pinned `requirements.txt` (sklearn, xgboost, lightgbm,
imbalanced-learn, shap, optuna, streamlit, pandas, matplotlib, seaborn,
scipy; pin to the legacy floor in the orchestration doc). Create the directory
tree, `__init__.py` files, `.gitignore` (`.venv/`, `data/raw/`,
`data/processed/`, `results/models/`). **Gate:** `pip install` + `python -c
"import xgboost, lightgbm, imblearn, shap, optuna, streamlit"` must succeed,
then `pip freeze` → pin. BLOCKED if any import fails.
**Verify:** imports succeed; tree matches spec.

### Task 1 — `data/download.py`  ·  model: haiku
Dataset 1: IBM Telco CSV from the raw GitHub URL (7,043 rows, 21 cols, ~26.5%
churn). Dataset 2: e-commerce churn via kaggle API *most-downloaded*; if
unavailable, fall back to a second Telco split + synthetic features (document
which path ran). Save to `data/raw/`. Coerce `TotalCharges` to numeric
(blank → NaN), strip whitespace.
**Verify:** files land in `data/raw/`; Telco shape == (7043, 21); churn rate in
[0.25, 0.28].

### Task 2 — `features/engineering.py`  ·  model: sonnet
Derived features per spec: tenure (`tenure_bucket` early/mid/loyal,
`is_new_customer`, `tenure_squared`); spend (`monthly_charge_per_service`,
`total_vs_expected`, charge-increase proxy); service (`num_services`,
`has_support`, `is_bundle`); contract risk (`is_month_to_month`,
`payment_auto`, `risk_score`); interactions (`tenure*monthly`,
`is_month_to_month*is_new_customer`). Pure function `engineer(df)->df`, no leak
of target.
**Verify:** smoke test asserts each new column exists, no NaN explosion, no use
of the churn column as an input.

### Task 3 — `features/selection.py`  ·  model: sonnet
Mutual information vs target; correlation matrix flagging `|r|>0.9`; RF
importances. Select top-N per method; return **union of top-20**.
**Verify:** returns a deduped feature list; redundant pairs reported.

### Task 4 — `features/preprocessing.py`  ·  model: sonnet
OrdinalEncoder for ordinals, OneHotEncoder for nominals, StandardScaler for
numerics, assembled as a sklearn `ColumnTransformer`. Stratified 70/15/15
split. Return Pipeline-compatible transformers (fit on train only — no leak).
**Verify:** split ratios within ±1%; transformer fit on train, applied to
val/test; output has no NaN.

### Task 5 — `imbalance/handler.py`  ·  model: sonnet
Four strategies returning a uniform interface: (1) none, (2) class weights
(`class_weight='balanced'` / `scale_pos_weight`), (3) SMOTE
(imbalanced-learn, fit on **train only**), (4) threshold tuning maximizing F1
and the 5×FN cost metric. Evaluate each on val: precision, recall, F1,
AUC-ROC, **AUC-PR**.
**Verify:** SMOTE never touches val/test; threshold search returns a threshold
in (0,1); comparison table emitted.

### Task 6 — `models/baseline.py`  ·  model: haiku
Logistic Regression, L2, `class_weight` configurable. Interpretable floor.
**Verify:** fits on processed train, predicts proba on val, AUC-ROC > 0.5.

### Task 7 — `models/tree_models.py`  ·  model: sonnet
RandomForest (`n_estimators=300`, depth via CV), XGBoost
(`lr=0.05, n_estimators=500`, early stopping on val), LightGBM (same CV folds).
Common `fit/predict_proba` interface.
**Verify:** each model trains and beats the baseline AUC-PR on val.

### Task 8 — `models/trainer.py`  ·  model: sonnet
5-fold stratified CV; **Optuna** Bayesian tuning, 50 trials/model, optimize
**AUC-PR**. Save best per algorithm to `results/models/`.
**Verify:** study runs (reduce to 3 trials in smoke mode); best params + model
artifact saved per algo.

### Task 9 — `models/evaluator.py`  ·  model: sonnet
Held-out test metrics (AUC-ROC, AUC-PR, F1, precision, recall, accuracy, log
loss, Brier). Plots: overlaid ROC, overlaid PR, confusion matrix (best),
calibration curve.
**Verify:** metrics dict has all 8 keys per model; 4 figures written to
`results/`.

### Task 10 — `pipeline/full_pipeline.py`  ·  model: sonnet
Single sklearn `Pipeline` (preprocessing → imbalance → best estimator)
runnable end-to-end on raw input.
**Verify:** `fit` then `predict` on raw rows with no manual preprocessing.

### Task 11 — `explainability/shap_analysis.py`  ·  model: sonnet
On best model: global (beeswarm, mean-|SHAP| bar, dependence for top-3); local
waterfalls for 3 customers (true churner, true retained, a false negative).
Derive business-insight strings from SHAP magnitudes.
**Verify:** SHAP values shape == (n, features); 6+ figures saved; insight
strings reference real top features.

### Task 12 — `visualisation/plots.py`  ·  model: haiku
Reusable plot helpers shared by evaluator/SHAP/dashboard (consistent styling).
**Verify:** each helper returns a Figure and saves on demand.

### Task 13 — `dashboard/app.py` (Streamlit)  ·  model: sonnet
Tab 1 Model Comparison (ROC/PR + metrics table, best highlighted); Tab 2
Predict a Customer (input form → probability gauge + SHAP waterfall +
suggested intervention); Tab 3 Segment Analysis (churn by contract/tenure/
service count + highest-risk segment + SHAP summary).
**Verify:** `streamlit run dashboard/app.py` boots headless; loads saved model;
a sample prediction renders.

### Task 14 — `results/summary.txt` + README  ·  model: haiku
Generate `summary.txt` (dataset stats, model comparison table on test, top SHAP
churn drivers). README: overview, dataset stats, comparison table, embedded
SHAP summary plot, business insights, install/run/dashboard commands.
**Verify:** summary fields populated from real run; README commands match the
actual entrypoints.

---

## Final whole-branch review (model: opus)
Spec coverage of all 4 imbalance strategies, AUC-PR as the tuning target, the
5×FN cost in threshold logic, no train/val leakage (SMOTE & scaler fit on train
only), SHAP on the selected best model, dashboard loads a persisted model.
Then `superpowers:finishing-a-development-branch`.
