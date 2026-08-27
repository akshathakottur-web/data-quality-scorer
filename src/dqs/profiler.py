"""
profiler.py

Responsible for loading a CSV file and generating a structural profile
of the dataset: shape, dtypes, memory usage, basic statistics, and
a best-guess at the target/label column.
"""

import re
import pandas as pd
import numpy as np

_ID_NAME_PATTERN = re.compile(r'(^id$|_id$|^idx$|^index$|^uuid$|^guid$)', re.IGNORECASE)


def _coerce_numeric_like_columns(df: pd.DataFrame, min_numeric_ratio: float = 0.8) -> pd.DataFrame:
    """
    Some columns are read as text purely because one or two bad values
    (e.g. 'unknown', 'N/A') sit alongside otherwise-numeric data. If
    left as text, those columns are silently excluded from every
    numeric detector (outliers, anomalies, correlation, etc).

    If a column is mostly numeric (>= min_numeric_ratio of its non-null
    values parse as numbers), convert the whole column to numeric and
    let the unparseable entries become NaN — which then correctly shows
    up as a missing value instead of disappearing entirely.
    """
    df = df.copy()
    for col in df.columns:
        # Skip columns that are already numeric or datetime.
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.notna().mean() >= min_numeric_ratio:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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
        The loaded dataset, with mostly-numeric text columns coerced
        to true numeric dtype (see _coerce_numeric_like_columns).
    """
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        # Fallback for files with non-UTF8 encoding
        df = pd.read_csv(filepath, encoding="latin1")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    df = _coerce_numeric_like_columns(df)

    return df


def detect_id_like_columns(df: pd.DataFrame) -> list:
    """
    Identify columns that are identifiers rather than real features
    (e.g. a primary-key style 'id' column). These should be excluded
    from numeric statistics (outliers, anomalies, correlation) and
    from duplicate-row checks, since a unique ID on every row makes
    every row look "different" even when every other field is an
    exact duplicate.

    A column counts as ID-like if either:
      - its name matches a common ID naming pattern (id, user_id,
        idx, index, uuid, guid), or
      - it's a fully-populated integer column where every value is
        unique AND the values form a contiguous sequence (i.e. it
        looks like an auto-increment key, not a real measurement).

    Returns
    -------
    list of column names.
    """
    id_like = []
    n_rows = len(df)

    for col in df.columns:
        name_match = bool(_ID_NAME_PATTERN.search(str(col).strip()))

        looks_sequential = False
        series = df[col]
        if pd.api.types.is_integer_dtype(series) and series.notna().all() and n_rows > 1:
            if series.nunique() == n_rows:
                sorted_vals = sorted(series.tolist())
                if sorted_vals == list(range(sorted_vals[0], sorted_vals[0] + n_rows)):
                    looks_sequential = True

        if name_match or looks_sequential:
            id_like.append(col)

    return id_like


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
        per-column stats, ID-like columns, and the guessed target column.
    """
    n_rows, n_cols = df.shape

    id_like_cols = detect_id_like_columns(df)

    # Exclude ID-like columns from numeric_cols -- they're identifiers,
    # not features, and including them dilutes/distorts every numeric
    # detector downstream (outliers, anomalies, correlation).
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in id_like_cols
    ]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    ]
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
            "is_id_like": col in id_like_cols,
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
        "id_like_cols": id_like_cols,
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
    print(f"ID-like columns (excluded from stats): {profile['id_like_cols']}")
    print(f"Guessed target column: {profile['guessed_target']}")
    print("-" * 50)
    for col, info in profile["column_profiles"].items():
        print(f"{col}: dtype={info['dtype']}, missing={info['pct_missing']}%, unique={info['n_unique']}")
    print("=" * 50)