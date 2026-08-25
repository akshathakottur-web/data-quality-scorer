"""
fixes.py

A fixed set of safe, well-tested dataset-fixing operations.
The chatbot can only call these functions — never arbitrary code —
which keeps fixes safe, reversible, and auditable.
"""

import pandas as pd


def drop_columns(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, str]:
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    new_df = df.drop(columns=existing)
    msg = f"Dropped columns: {existing}."
    if missing:
        msg += f" (Not found, skipped: {missing})"
    return new_df, msg


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    before = len(df)
    new_df = df.drop_duplicates()
    removed = before - len(new_df)
    return new_df, f"Removed {removed} duplicate rows."


def impute_missing(df: pd.DataFrame, column: str, strategy: str = "median") -> tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Column '{column}' not found."

    new_df = df.copy()
    n_missing = new_df[column].isna().sum()

    if strategy == "median" and pd.api.types.is_numeric_dtype(new_df[column]):
        fill_value = new_df[column].median()
    elif strategy == "mean" and pd.api.types.is_numeric_dtype(new_df[column]):
        fill_value = new_df[column].mean()
    elif strategy == "mode":
        fill_value = new_df[column].mode().iloc[0] if not new_df[column].mode().empty else None
    else:
        fill_value = new_df[column].mode().iloc[0] if not new_df[column].mode().empty else None

    new_df[column] = new_df[column].fillna(fill_value)
    return new_df, f"Filled {n_missing} missing values in '{column}' using {strategy} ({fill_value})."


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Column '{column}' not found."

    new_df = df.copy()
    q1 = new_df[column].quantile(0.25)
    q3 = new_df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    n_capped = ((new_df[column] < lower) | (new_df[column] > upper)).sum()
    new_df[column] = new_df[column].clip(lower, upper)
    return new_df, f"Capped {n_capped} outliers in '{column}' to range [{round(lower,2)}, {round(upper,2)}]."


def remove_rows_by_index(df: pd.DataFrame, indices: list) -> tuple[pd.DataFrame, str]:
    valid_idx = [i for i in indices if i in df.index]
    new_df = df.drop(index=valid_idx)
    return new_df, f"Removed {len(valid_idx)} rows by index."


# Registry the chatbot is allowed to call — name -> function.
# This is the ONLY set of operations the LLM can trigger.
AVAILABLE_FIXES = {
    "drop_columns": drop_columns,
    "drop_duplicate_rows": drop_duplicate_rows,
    "impute_missing": impute_missing,
    "cap_outliers_iqr": cap_outliers_iqr,
    "remove_rows_by_index": remove_rows_by_index,
}