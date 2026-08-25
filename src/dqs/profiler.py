"""
profiler.py

Responsible for loading a CSV file and generating a structural profile
of the dataset: shape, dtypes, memory usage, basic statistics, and
a best-guess at the target/label column.
"""

import pandas as pd
import numpy as np


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with safe defaults.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        # Fallback for files with non-UTF8 encoding
        df = pd.read_csv(filepath, encoding="latin1")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    return df


def guess_target_column(df: pd.DataFrame) -> str | None:
    """
    Attempt to guess which column is the target/label column.

    Heuristic: look for common target column names first.
    If none found, fall back to the last column IF it looks
    categorical (few unique values relative to row count).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str or None
        The guessed target column name, or None if no reasonable guess.
    """
    common_names = [
        "target", "label", "class", "y",
        "survived", "outcome", "churn", "default"
    ]

    for col in df.columns:
        if col.strip().lower() in common_names:
            return col

    # Fallback: check if last column looks categorical
    last_col = df.columns[-1]
    n_unique = df[last_col].nunique(dropna=True)
    if n_unique <= 20 and n_unique < len(df) * 0.5:
        return last_col

    return None


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a structural profile of the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        A dictionary containing shape, dtypes, memory usage,
        per-column stats, and the guessed target column.
    """
    n_rows, n_cols = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    column_profiles = {}
    for col in df.columns:
        col_data = df[col]
        col_info = {
            "dtype": str(col_data.dtype),
            "n_missing": int(col_data.isna().sum()),
            "pct_missing": round(float(col_data.isna().mean() * 100), 2),
            "n_unique": int(col_data.nunique(dropna=True)),
        }

        if col in numeric_cols:
            col_info.update({
                "mean": round(float(col_data.mean()), 4) if not col_data.dropna().empty else None,
                "std": round(float(col_data.std()), 4) if not col_data.dropna().empty else None,
                "min": float(col_data.min()) if not col_data.dropna().empty else None,
                "max": float(col_data.max()) if not col_data.dropna().empty else None,
            })

        column_profiles[col] = col_info

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "memory_usage_mb": round(memory_usage_mb, 3),
        "column_profiles": column_profiles,
        "guessed_target": guess_target_column(df),
    }

    return profile


def print_profile_summary(profile: dict) -> None:
    """
    Pretty-print a profile dictionary to the console for quick inspection.
    """
    print("=" * 50)
    print("DATASET PROFILE SUMMARY")
    print("=" * 50)
    print(f"Rows: {profile['n_rows']}")
    print(f"Columns: {profile['n_cols']}")
    print(f"Memory usage: {profile['memory_usage_mb']} MB")
    print(f"Numeric columns: {profile['numeric_cols']}")
    print(f"Categorical columns: {profile['categorical_cols']}")
    print(f"Guessed target column: {profile['guessed_target']}")
    print("-" * 50)
    for col, info in profile["column_profiles"].items():
        print(f"{col}: dtype={info['dtype']}, missing={info['pct_missing']}%, unique={info['n_unique']}")
    print("=" * 50)