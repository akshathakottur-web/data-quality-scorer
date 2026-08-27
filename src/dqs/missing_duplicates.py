"""
missing_duplicates.py

Detects missing values and duplicate rows, and scores the dataset
accordingly.
"""

import pandas as pd


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyze missing values column by column.

    Returns
    -------
    dict with:
        - total_missing_cells
        - pct_missing_overall
        - column_missing: per-column % missing
        - high_missing_columns: columns above 40% missing (risky)
        - score: 0-100, higher is better
    """
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    pct_missing_overall = round((total_missing / total_cells) * 100, 2) if total_cells else 0.0

    column_missing = {
        col: round(float(df[col].isna().mean() * 100), 2)
        for col in df.columns
    }

    high_missing_columns = [
        col for col, pct in column_missing.items() if pct > 40.0
    ]

    # Scoring: penalize based on overall missingness, extra penalty
    # for columns that are almost entirely empty.
    score = 100 - min(pct_missing_overall * 2, 60)
    score -= len(high_missing_columns) * 10
    score = max(0, round(score, 1))

    return {
        "total_missing_cells": total_missing,
        "pct_missing_overall": pct_missing_overall,
        "column_missing": column_missing,
        "high_missing_columns": high_missing_columns,
        "score": score,
    }


def analyze_duplicates(df: pd.DataFrame, id_like_cols: list = None) -> dict:
    """
    Analyze exact duplicate rows.

    Parameters
    ----------
    id_like_cols : list, optional
        Columns to exclude before checking for duplicates (e.g. a
        primary-key 'id' column). Without this, a unique ID on every
        row makes every row look "different" even when every other
        field is an exact duplicate -- so real duplicates go undetected.
        Pass profile['id_like_cols'] from profiler.profile_dataset here.

    Returns
    -------
    dict with:
        - n_duplicate_rows
        - pct_duplicate_rows
        - duplicate_row_indices (sample, max 20)
        - score: 0-100, higher is better
    """
    n_rows = len(df)

    check_df = df.drop(columns=id_like_cols or [], errors="ignore")

    duplicate_mask = check_df.duplicated(keep="first")
    n_duplicates = int(duplicate_mask.sum())
    pct_duplicates = round((n_duplicates / n_rows) * 100, 2) if n_rows else 0.0

    duplicate_indices = df[duplicate_mask].index.tolist()[:20]

    score = 100 - min(pct_duplicates * 3, 100)
    score = max(0, round(score, 1))

    return {
        "n_duplicate_rows": n_duplicates,
        "pct_duplicate_rows": pct_duplicates,
        "duplicate_row_indices": duplicate_indices,
        "score": score,
    }