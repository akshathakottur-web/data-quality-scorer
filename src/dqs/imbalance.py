"""
imbalance.py

Detects class imbalance in the target/label column.
"""

import pandas as pd


def analyze_class_imbalance(df: pd.DataFrame, target_col: str | None) -> dict:
    """
    Analyze class balance of the target column, if one is identified
    and looks like a classification target.

    Returns
    -------
    dict with class counts, imbalance ratio, and score.
    """
    if target_col is None or target_col not in df.columns:
        return {
            "applicable": False,
            "reason": "No target column identified.",
            "score": 100.0,
        }

    series = df[target_col].dropna()
    n_unique = series.nunique()

    # Heuristic: if too many unique values, this is likely regression,
    # not classification, so imbalance doesn't apply.
    if n_unique > 20 or n_unique / len(series) > 0.5:
        return {
            "applicable": False,
            "reason": "Target appears continuous (regression-like), not categorical.",
            "score": 100.0,
        }

    class_counts = series.value_counts().to_dict()
    majority_count = max(class_counts.values())
    minority_count = min(class_counts.values())
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count > 0 else float("inf")

    class_percentages = {
        str(k): round((v / len(series)) * 100, 2) for k, v in class_counts.items()
    }

    # Scoring: ratio of 1 (perfectly balanced) = 100.
    # Ratio of 10+ = heavily penalized.
    if imbalance_ratio <= 1.5:
        score = 100
    elif imbalance_ratio <= 3:
        score = 85
    elif imbalance_ratio <= 10:
        score = 60
    elif imbalance_ratio <= 50:
        score = 30
    else:
        score = 10

    return {
        "applicable": True,
        "target_col": target_col,
        "n_classes": n_unique,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "imbalance_ratio": imbalance_ratio,
        "score": score,
    }