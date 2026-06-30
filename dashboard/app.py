"""Streamlit dashboard for customer churn prediction.

Three tabs:
  1. Model Comparison  — metrics table + ROC/PR curves, best model highlighted
  2. Predict a Customer — input form → probability gauge + SHAP + intervention
  3. Segment Analysis  — churn by contract / tenure bucket / service count
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

RANDOM_SEED = 42
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
RAW_CSV = Path("data/raw/telco_churn.csv")

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_raw_df() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(subset=["TotalCharges"])


@st.cache_resource
def load_pipeline() -> tuple[np.ndarray, np.ndarray, Any, list[str], Any]:
    """Return (X_test_arr, y_test, transformer, feature_names, df_engineered)."""
    from features.engineering import engineer
    from features.preprocessing import fit_transform_splits, split_data
    from features.selection import select_features

    df = load_raw_df()
    df_eng = engineer(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_eng)
    feature_names, _ = select_features(df_eng)
    X_train_arr, _, X_test_arr, transformer = fit_transform_splits(
        X_train, X_val, X_test, feature_names
    )
    return X_test_arr, y_test, transformer, feature_names, df_eng


@st.cache_data
def load_metrics() -> dict:
    from models.evaluator import evaluate_all_models

    X_test_arr, y_test, *_ = load_pipeline()
    return evaluate_all_models(X_test_arr, y_test)


@st.cache_data
def get_best_name_and_auc() -> tuple[str, float]:
    from models.evaluator import find_best_model

    metrics = load_metrics()
    best_name, _ = find_best_model(metrics)
    return best_name, metrics[best_name]["auc_pr"]


@st.cache_data
def compute_modal_values() -> dict:
    """Modal (most-frequent or median) value for each raw column."""
    df = load_raw_df()
    modals: dict = {}
    for col in df.columns:
        if df[col].dtype == object:
            modals[col] = df[col].mode()[0]
        else:
            modals[col] = df[col].median()
    return modals


# ---------------------------------------------------------------------------
# Helper: build a single-row raw DataFrame
# ---------------------------------------------------------------------------

_ALL_PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def build_single_row(
    tenure: int,
    monthly_charges: float,
    contract: str,
    internet_service: str,
    tech_support: str,
    payment_method: str,
) -> pd.DataFrame:
    """Return a one-row DataFrame with all Telco columns filled."""
    modals = compute_modal_values()
    row = dict(modals)
    # Override with user inputs
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges
    row["Contract"] = contract
    row["InternetService"] = internet_service
    row["TechSupport"] = tech_support
    row["PaymentMethod"] = payment_method
    # Remove target if present
    row.pop("Churn", None)
    row.pop("customerID", None)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def predict_customer(row_df: pd.DataFrame) -> tuple[float, np.ndarray, list[str]]:
    """Predict churn probability for a single raw row.

    Returns:
        (prob, shap_values_1d, display_feature_names)
    """
    from features.engineering import engineer
    from models.evaluator import find_best_model

    _, _, transformer, feature_names, _ = load_pipeline()
    metrics = load_metrics()
    best_name, best_model = find_best_model(metrics)

    # Add dummy target so engineer() / feature selection doesn't fail
    row_df = row_df.copy()
    row_df["Churn"] = "No"

    df_eng = engineer(row_df)
    X_arr = transformer.transform(df_eng[feature_names])

    prob = float(best_model.predict_proba(X_arr)[0])

    raw_names = list(transformer.get_feature_names_out())
    display_names = [
        n.replace("numeric__", "").replace("nominal__", "").replace("ordinal__", "")
        for n in raw_names
    ]
    return prob, X_arr, display_names


# ---------------------------------------------------------------------------
# Risk intervention text
# ---------------------------------------------------------------------------


def intervention_text(prob: float) -> tuple[str, str]:
    if prob > 0.7:
        return "High Risk", (
            "Immediate action recommended: offer a contract upgrade discount, "
            "assign a customer success manager, and review service pain-points."
        )
    elif prob > 0.3:
        return "Medium Risk", (
            "Proactive retention: send a personalised offer, check satisfaction "
            "score, and highlight under-used service features."
        )
    else:
        return "Low Risk", "Customer appears stable. Continue standard engagement."


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "log_loss": "Log-Loss",
    "brier_score": "Brier",
}


def build_metrics_df(metrics: dict, best_name: str) -> pd.DataFrame:
    rows = []
    for algo, m in metrics.items():
        label = f"★ {algo}" if algo == best_name else algo
        rows.append({"Model": label, **{_METRIC_DISPLAY[k]: round(v, 4) for k, v in m.items() if k in _METRIC_DISPLAY}})
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["Model Comparison", "Predict a Customer", "Segment Analysis"])

# ===== TAB 1: Model Comparison =====
with tab1:
    st.header("Model Comparison")

    metrics = load_metrics()
    best_name, best_auc_pr = get_best_name_and_auc()

    st.success(f"Best model: **{best_name}** — AUC-PR = {best_auc_pr:.4f}")

    mdf = build_metrics_df(metrics, best_name)
    st.dataframe(mdf.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    col1, col2 = st.columns(2)
    roc_path = RESULTS_DIR / "roc_curves.png"
    pr_path = RESULTS_DIR / "pr_curves.png"

    with col1:
        st.subheader("ROC Curves")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.warning("roc_curves.png not found in results/")

    with col2:
        st.subheader("PR Curves")
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)
        else:
            st.warning("pr_curves.png not found in results/")

# ===== TAB 2: Predict a Customer =====
with tab2:
    st.header("Predict a Customer")

    with st.form("customer_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0, step=0.5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            payment_method = st.selectbox("Payment Method", _ALL_PAYMENT_METHODS)

        submitted = st.form_submit_button("Predict Churn Probability")

    if submitted:
        row_df = build_single_row(
            tenure, monthly_charges, contract, internet_service, tech_support, payment_method
        )
        try:
            prob, _, display_names = predict_customer(row_df)

            risk_label, advice = intervention_text(prob)
            risk_color = {"High Risk": "error", "Medium Risk": "warning", "Low Risk": "success"}[risk_label]

            st.metric("Churn Probability", f"{prob:.1%}")
            getattr(st, risk_color)(f"**{risk_label}** — {advice}")

            # SHAP waterfall: use pre-generated image for churner sample
            shap_path = RESULTS_DIR / "shap_waterfall_churner.png"
            if shap_path.exists():
                st.subheader("SHAP Explanation (sample churner)")
                st.image(str(shap_path), use_container_width=True)
            else:
                st.info("SHAP waterfall image not available.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ===== TAB 3: Segment Analysis =====
with tab3:
    st.header("Segment Analysis")

    _, _, _, _, df_eng = load_pipeline()

    def segment_churn_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return (
            df.groupby(col)["Churn"]
            .apply(lambda s: (s.map({"Yes": 1, "No": 0}).fillna(s).astype(int).mean()))
            .reset_index()
            .rename(columns={"Churn": "churn_rate"})
            .sort_values("churn_rate", ascending=False)
        )

    seg_contract = segment_churn_rate(df_eng, "Contract")
    seg_tenure = segment_churn_rate(df_eng, "tenure_bucket")
    seg_services = segment_churn_rate(df_eng, "num_services")

    def highlight_max(series: pd.Series) -> list[str]:
        max_val = series.max()
        return ["background-color: #f8d7da" if v == max_val else "" for v in series]

    for title, seg_df, col in [
        ("Churn Rate by Contract Type", seg_contract, "Contract"),
        ("Churn Rate by Tenure Bucket", seg_tenure, "tenure_bucket"),
        ("Churn Rate by Number of Services", seg_services, "num_services"),
    ]:
        st.subheader(title)
        highest = seg_df.iloc[0][col]
        st.caption(f"Highest-risk segment: **{highest}** ({seg_df.iloc[0]['churn_rate']:.1%})")
        styled = (
            seg_df.style
            .apply(highlight_max, subset=["churn_rate"])
            .format({"churn_rate": "{:.2%}"})
        )
        st.dataframe(styled, use_container_width=True)
        st.bar_chart(seg_df.set_index(col)["churn_rate"])

    st.subheader("SHAP Feature Importance")
    shap_bar_path = RESULTS_DIR / "shap_bar.png"
    if shap_bar_path.exists():
        st.image(str(shap_bar_path), use_container_width=True)
    else:
        st.warning("shap_bar.png not found in results/")


if __name__ == "__main__":
    print("dashboard.app smoke test: OK")
