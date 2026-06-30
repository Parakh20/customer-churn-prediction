"""Reusable plot helpers for Streamlit dashboard and evaluation reports.

Each helper returns a matplotlib Figure object and optionally saves to disk.
All plots use consistent styling via _apply_style() helper.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Use non-interactive backend (no display server needed)
matplotlib.use("Agg")

RANDOM_SEED = 42


def _apply_style(ax: matplotlib.axes.Axes) -> None:
    """Apply consistent styling to plot axes.

    Sets clean grid (y-axis only, light gray), removes top/right spines,
    and applies standardized font sizes.

    Args:
        ax: Matplotlib axis to style.
    """
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)


def plot_churn_distribution(y: np.ndarray, save_path: str | None = None) -> Figure:
    """Bar chart of churn vs. not-churn counts with percentages.

    Args:
        y: Binary target array (0=no churn, 1=churn).
        save_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = np.bincount(y.astype(int))
    labels = ["No Churn", "Churn"]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add count and percentage labels on bars
    total = counts.sum()
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Churn Distribution", fontsize=13, fontweight="bold")
    _apply_style(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def plot_feature_importance(
    feature_names: list[str] | np.ndarray,
    importances: np.ndarray,
    title: str = "Feature Importances",
    top_n: int = 20,
    save_path: str | None = None,
) -> Figure:
    """Horizontal bar chart of top-N feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values.
        title: Plot title.
        top_n: Number of top features to display.
        save_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Sort and select top-N
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))

    ax.barh(sorted_names, sorted_importances, color="#3498db", alpha=0.8, edgecolor="black", linewidth=1)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    _apply_style(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def _draw_gauge_arcs(ax: matplotlib.axes.Axes, radius: float = 1) -> None:
    """Draw gauge arcs and color zones for probability gauge.

    Args:
        ax: Matplotlib axis to draw on.
        radius: Radius of the gauge arc.
    """
    theta = np.linspace(np.pi, 0, 100)

    # Background arc (full range)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", linewidth=2)

    # Color zones: green (0-0.5), yellow (0.5-0.75), red (0.75-1.0)
    zones = [
        (np.pi, np.pi * 0.75, "#2ecc71"),  # Green: 0-0.5
        (np.pi * 0.75, np.pi * 0.5, "#f39c12"),  # Yellow: 0.5-0.75
        (np.pi * 0.5, np.pi * 0.25, "#e74c3c"),  # Red: 0.75-1.0
    ]

    for start, end, color in zones:
        zone_theta = np.linspace(start, end, 50)
        ax.fill_between(radius * np.cos(zone_theta), radius * np.sin(zone_theta), 0, alpha=0.3, color=color)
        ax.plot(radius * np.cos(zone_theta), radius * np.sin(zone_theta), color=color, linewidth=3)


def _draw_gauge_needle(ax: matplotlib.axes.Axes, churn_prob: float) -> None:
    """Draw needle and labels on probability gauge.

    Args:
        ax: Matplotlib axis to draw on.
        churn_prob: Churn probability between 0 and 1.
    """
    # Needle pointing to current probability
    angle = np.pi - (churn_prob * np.pi)  # Convert prob (0-1) to angle
    needle_length = 0.8
    ax.arrow(
        0,
        0,
        needle_length * np.cos(angle),
        needle_length * np.sin(angle),
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
        linewidth=2,
    )

    # Scale labels
    ax.text(1.1, -0.2, "0%", ha="center", fontsize=9)
    ax.text(0.65, 0.8, "50%", ha="center", fontsize=9)
    ax.text(-1.1, -0.2, "100%", ha="center", fontsize=9)

    # Central text
    ax.text(0, -0.4, f"{100*churn_prob:.1f}%", ha="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.55, "Churn Probability", ha="center", fontsize=11)


def plot_probability_gauge(churn_prob: float, save_path: str | None = None) -> Figure:
    """Single-value gauge/dial showing churn probability (0–1).

    Args:
        churn_prob: Churn probability between 0 and 1.
        save_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    _draw_gauge_arcs(ax)
    _draw_gauge_needle(ax, churn_prob)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.7, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def _style_metrics_table(table: matplotlib.table.Table, df: pd.DataFrame, best_model: str) -> None:
    """Style metrics table with header and best model highlighting.

    Args:
        table: Matplotlib table object to style.
        df: DataFrame with metrics (used to get column/row counts).
        best_model: Name of the best model to highlight.
    """
    # Highlight best model row (cell indexing: (row, col) where row 0 is header)
    best_row_idx = list(df.index).index(best_model) + 1
    for col in range(len(df.columns) + 1):
        if (best_row_idx, col) in table._cells:
            cell = table[(best_row_idx, col)]
            cell.set_facecolor("#fff3cd")
            cell.set_text_props(weight="bold")

    # Style header row
    for col in range(len(df.columns) + 1):
        if (0, col) in table._cells:
            cell = table[(0, col)]
            cell.set_facecolor("#3498db")
            cell.set_text_props(weight="bold", color="white")


def plot_metrics_table(
    metrics_dict: dict[str, dict[str, float]], save_path: str | None = None
) -> Figure:
    """Table figure showing all models and their 8 metrics, best model highlighted.

    Args:
        metrics_dict: Nested dict {algo_name: {metric_name: value}}.
        save_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T

    # Identify best model by auc_pr (first column after sorting)
    if "auc_pr" in df.columns:
        best_model = df["auc_pr"].idxmax()
    else:
        best_model = df.index[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
        colWidths=[0.12] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    _style_metrics_table(table, df, best_model)

    ax.set_title("Model Metrics Comparison", fontsize=13, fontweight="bold", pad=20)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def plot_segment_churn(
    df: pd.DataFrame, segment_col: str, target_col: str = "Churn", save_path: str | None = None
) -> Figure:
    """Bar chart of churn rate by segment (e.g., by Contract type, tenure bucket).

    Args:
        df: DataFrame containing data.
        segment_col: Column name to segment by.
        target_col: Column name for churn target (default: "Churn").
        save_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Calculate churn rate by segment
    segment_data = df.groupby(segment_col)[target_col].agg(["sum", "count"])
    segment_data["churn_rate"] = (segment_data["sum"] / segment_data["count"]) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    segments = segment_data.index.astype(str)
    churn_rates = segment_data["churn_rate"].values
    counts = segment_data["count"].values

    bars = ax.bar(segments, churn_rates, color="#9b59b6", alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add count labels on bars
    for bar, count, rate in zip(bars, counts, churn_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{rate:.1f}%\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(segment_col, fontsize=11)
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_title(f"Churn Rate by {segment_col}", fontsize=13, fontweight="bold")
    _apply_style(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    # Smoke test: generate all 5 plots with synthetic data
    np.random.seed(RANDOM_SEED)

    # Test plot_churn_distribution
    y_test = np.random.binomial(1, 0.3, 100)
    fig1 = plot_churn_distribution(y_test)
    assert isinstance(fig1, Figure), "plot_churn_distribution must return Figure"
    plt.close(fig1)

    # Test plot_feature_importance
    feature_names = [f"Feature_{i}" for i in range(15)]
    importances = np.random.rand(15)
    fig2 = plot_feature_importance(feature_names, importances, top_n=10)
    assert isinstance(fig2, Figure), "plot_feature_importance must return Figure"
    plt.close(fig2)

    # Test plot_probability_gauge
    fig3 = plot_probability_gauge(0.65)
    assert isinstance(fig3, Figure), "plot_probability_gauge must return Figure"
    plt.close(fig3)

    # Test plot_metrics_table
    metrics_dict = {
        "LogisticRegression": {
            "auc_roc": 0.75,
            "auc_pr": 0.68,
            "f1": 0.70,
            "precision": 0.72,
            "recall": 0.68,
            "accuracy": 0.82,
            "log_loss": 0.45,
            "brier_score": 0.15,
        },
        "RandomForest": {
            "auc_roc": 0.82,
            "auc_pr": 0.76,
            "f1": 0.74,
            "precision": 0.75,
            "recall": 0.73,
            "accuracy": 0.85,
            "log_loss": 0.38,
            "brier_score": 0.12,
        },
    }
    fig4 = plot_metrics_table(metrics_dict)
    assert isinstance(fig4, Figure), "plot_metrics_table must return Figure"
    plt.close(fig4)

    # Test plot_segment_churn
    df_test = pd.DataFrame(
        {"Churn": np.random.binomial(1, 0.3, 100), "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100)}
    )
    fig5 = plot_segment_churn(df_test, "Contract")
    assert isinstance(fig5, Figure), "plot_segment_churn must return Figure"
    plt.close(fig5)

    # Test save functionality
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fig_save = plot_churn_distribution(y_test, save_path=str(tmp_path / "test_churn.png"))
        assert (tmp_path / "test_churn.png").exists(), "save_path must write file"
        plt.close(fig_save)

    print("visualisation.plots smoke test: OK")
