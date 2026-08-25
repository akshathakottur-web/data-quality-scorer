"""
report.py

Generates a full text report and visualizations summarizing
data quality findings.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_text_report(profile: dict, all_reports: dict,
                          scoring: dict, recommendations: list) -> str:
    """
    Build a full plain-text report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AUTOMATIC DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Rows: {profile['n_rows']}  |  Columns: {profile['n_cols']}")
    lines.append(f"Guessed target column: {profile['guessed_target']}")
    lines.append("")
    lines.append(f"OVERALL DATA QUALITY SCORE: {scoring['overall_score']}/100")
    lines.append(f"STATUS: {scoring['status']}")
    lines.append("-" * 60)
    lines.append("INDIVIDUAL SCORES:")
    for key, score in scoring["individual_scores"].items():
        lines.append(f"  - {key}: {score}/100")
    lines.append("-" * 60)
    lines.append("RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_text_report(report_text: str, output_path: str = "outputs/reports/report.txt") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Text report saved to: {output_path}")


def generate_visualizations(df: pd.DataFrame, profile: dict,
                             all_reports: dict,
                             output_dir: str = "outputs/reports") -> None:
    """
    Generate and save key visualizations: missing value heatmap,
    score breakdown bar chart, target distribution, and outlier analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="Reds")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values_heatmap.png")
    plt.close()

    # 2. Individual scores bar chart
    scores = all_reports.get("_individual_scores", {})
    if scores:
        plt.figure(figsize=(10, 5))
        names = list(scores.keys())
        values = list(scores.values())
        colors = ["#2ecc71" if v >= 75 else "#f39c12" if v >= 50 else "#e74c3c" for v in values]
        plt.barh(names, values, color=colors)
        plt.xlabel("Score (0-100)")
        plt.title("Data Quality Sub-Scores")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_breakdown.png")
        plt.close()

    # 3. Target distribution, if applicable
    target = profile.get("guessed_target")
    if target and target in df.columns:
        plt.figure(figsize=(6, 5))
        df[target].value_counts().plot(kind="bar", color="#3498db")
        plt.title(f"Target Distribution: {target}")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution.png")
        plt.close()

    # ---------------------------------------------
    # 4. OUTLIER BOXPLOTS
    # ---------------------------------------------

    numeric_cols = profile.get("numeric_cols", [])

    if numeric_cols:
        plt.figure(figsize=(12, 6))

        df[numeric_cols].boxplot(
            rot=45
        )

        plt.title("Outlier Analysis - Boxplots")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/outlier_boxplots.png")
        plt.close()

    # ---------------------------------------------
    # 5. OUTLIER PERCENTAGE BY COLUMN
    # ---------------------------------------------

    outlier_report = all_reports.get("outliers_iqr", {})
    column_outliers = outlier_report.get("column_outliers", {})

    if column_outliers:
        outlier_columns = list(column_outliers.keys())
        outlier_percentages = [
            column_outliers[col]["pct_outliers"]
            for col in outlier_columns
        ]

        plt.figure(figsize=(10, 5))

        plt.bar(
            outlier_columns,
            outlier_percentages
        )

        plt.xlabel("Column")
        plt.ylabel("Outliers (%)")
        plt.title("Percentage of Outliers by Column")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/outlier_percentage.png")
        plt.close()

    # ---------------------------------------------
    # 6. ISOLATION FOREST OUTLIERS
    # ---------------------------------------------

    iforest_report = all_reports.get("outliers_iforest", {})
    outlier_indices = iforest_report.get("outlier_indices", [])

    if len(numeric_cols) >= 2:

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        plt.figure(figsize=(8, 6))

        plt.scatter(
            df[x_col],
            df[y_col],
            alpha=0.5,
            label="Normal Data"
        )

        valid_outlier_indices = [
            index for index in outlier_indices
            if index in df.index
        ]

        if valid_outlier_indices:
            outlier_df = df.loc[valid_outlier_indices]

            plt.scatter(
                outlier_df[x_col],
                outlier_df[y_col],
                color="red",
                label="Isolation Forest Outliers"
            )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Isolation Forest Detected Outliers")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{output_dir}/isolation_forest_outliers.png")
        plt.close()

    # ---------------------------------------------
    # 7. HISTOGRAM FOR EACH NUMERIC FEATURE
    # ---------------------------------------------

    for col in numeric_cols:

        safe_name = str(col).replace(" ", "_").replace("/", "_")

        plt.figure(figsize=(8, 5))

        sns.histplot(
            df[col].dropna(),
            kde=True
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/histogram_{safe_name}.png"
        )

        plt.close()

    print(f"Visualizations saved to: {output_dir}/")