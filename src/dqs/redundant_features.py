"""
redundant_features.py

Detects constant columns and highly correlated (redundant) feature pairs.
"""

import pandas as pd
import numpy as np


def detect_constant_columns(df: pd.DataFrame) -> dict:
    """
    Detect columns with only one unique value (zero variance).

    Returns
    -------
    dict with list of constant columns and score.
    """
    constant_cols = []
    for col in df.columns:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_cols.append(col)

    n_total = len(df.columns) or 1
    score = 100 - (len(constant_cols) / n_total) * 100
    score = max(0, round(score, 1))

    return {
        "constant_columns": constant_cols,
        "score": score,
    }


def detect_correlated_features(df: pd.DataFrame, numeric_cols: list,
                                threshold: float = 0.9) -> dict:
    """
    Detect pairs of numeric features that are highly correlated.

    Parameters
    ----------
    threshold : float
        Absolute correlation value above which a pair is flagged
        as redundant (default 0.9).

    Returns
    -------
    dict with correlated pairs and score.
    """
    if len(numeric_cols) < 2:
        return {
            "correlated_pairs": [],
            "score": 100.0,
            "note": "Not enough numeric columns to check correlation.",
        }

    corr_matrix = df[numeric_cols].corr(numeric_only=True).abs()

    correlated_pairs = []
    seen = set()
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 == col2:
                continue
            pair_key = tuple(sorted([col1, col2]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            corr_value = corr_matrix.loc[col1, col2]
            if pd.notna(corr_value) and corr_value >= threshold:
                correlated_pairs.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(float(corr_value), 3),
                })

    score = 100 - min(len(correlated_pairs) * 15, 100)
    score = max(0, round(score, 1))

    return {
        "correlated_pairs": correlated_pairs,
        "score": score,
    }