"""
outliers.py

Detects outliers using two complementary methods:
1. IQR (statistical, per-column)
2. Isolation Forest (ML-based, multivariate)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Detect outliers per numeric column using the IQR method.

    Returns
    -------
    dict with per-column outlier counts and an overall score.
    """
    column_outliers = {}
    total_outlier_cells = 0

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = int(outlier_mask.sum())
        total_outlier_cells += n_outliers

        column_outliers[col] = {
            "n_outliers": n_outliers,
            "pct_outliers": round((n_outliers / len(series)) * 100, 2),
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
        }

    total_numeric_cells = sum(len(df[c].dropna()) for c in numeric_cols) or 1
    pct_overall = round((total_outlier_cells / total_numeric_cells) * 100, 2)

    score = 100 - min(pct_overall * 4, 100)
    score = max(0, round(score, 1))

    return {
        "column_outliers": column_outliers,
        "total_outlier_cells": total_outlier_cells,
        "pct_overall": pct_overall,
        "score": score,
    }


def detect_outliers_isolation_forest(df: pd.DataFrame, numeric_cols: list,
                                      contamination: float = 0.05) -> dict:
    """
    Detect multivariate outliers using Isolation Forest.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers in the dataset (default 5%).
        This is a hyperparameter we set based on domain assumption,
        not learned from data.

    Returns
    -------
    dict with outlier row indices, count, and score.
    """
    numeric_df = df[numeric_cols].copy()

    # Isolation Forest can't handle NaNs, so we fill them temporarily
    # with the column median just for this detection step.
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    if numeric_df.shape[1] == 0 or numeric_df.shape[0] < 10:
        return {
            "n_outliers": 0,
            "pct_outliers": 0.0,
            "outlier_indices": [],
            "score": 100.0,
            "note": "Not enough numeric data to run Isolation Forest.",
        }

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    predictions = model.fit_predict(numeric_df)  # -1 = outlier, 1 = normal

    outlier_mask = predictions == -1
    n_outliers = int(outlier_mask.sum())
    pct_outliers = round((n_outliers / len(df)) * 100, 2)

    outlier_indices = df.index[outlier_mask].tolist()[:20]

    score = 100 - min(pct_outliers * 3, 100)
    score = max(0, round(score, 1))

    return {
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "outlier_indices": outlier_indices,
        "score": score,
    }


# -------------------------------------------------
# OUTLIER VISUALIZATION FUNCTIONS
# -------------------------------------------------

def generate_outlier_visualizations(
    df: pd.DataFrame,
    numeric_cols: list,
    iqr_report: dict,
    iforest_report: dict,
    output_dir: str = "outputs/reports",
):
    """
    Generate graphs for detailed outlier analysis.
    """

    os.makedirs(output_dir, exist_ok=True)

    if not numeric_cols:
        return

    # ---------------------------------------------
    # 1. BOX PLOTS
    # ---------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    df[numeric_cols].boxplot(ax=ax, rot=45)

    ax.set_title("Box Plot of Numeric Features")
    ax.set_ylabel("Values")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "outlier_boxplots.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------
    # 2. OUTLIER PERCENTAGE BAR CHART
    # ---------------------------------------------

    outlier_data = iqr_report.get("column_outliers", {})

    if outlier_data:

        columns = list(outlier_data.keys())

        percentages = [
            outlier_data[col]["pct_outliers"]
            for col in columns
        ]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(columns, percentages)

        ax.set_title("Percentage of Outliers by Column")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Outliers (%)")

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, "outlier_percentage.png"),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------
    # 3. HISTOGRAMS
    # ---------------------------------------------

    for col in numeric_cols:

        series = df[col].dropna()

        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(series, bins=30)

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        plt.tight_layout()

        safe_name = str(col).replace(" ", "_").replace("/", "_")

        plt.savefig(
            os.path.join(
                output_dir,
                f"histogram_{safe_name}.png",
            ),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------
    # 4. ISOLATION FOREST ANOMALY VISUALIZATION
    # ---------------------------------------------

    if len(numeric_cols) >= 2:

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        outlier_indices = iforest_report.get(
            "outlier_indices",
            []
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            df[x_col],
            df[y_col],
            alpha=0.6,
            label="Normal Data",
        )

        if outlier_indices:

            outlier_points = df.loc[
                df.index.isin(outlier_indices)
            ]

            ax.scatter(
                outlier_points[x_col],
                outlier_points[y_col],
                marker="x",
                s=100,
                label="Isolation Forest Outliers",
            )

        ax.set_title(
            f"Isolation Forest Outliers: {x_col} vs {y_col}"
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "isolation_forest_outliers.png",
            ),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()