"""
row_quality.py

Computes a size-invariant overall data quality score based on what
FRACTION OF ROWS have at least one real problem, rather than averaging
independent per-category percentages together.

Why this matters: averaging separate detector scores (missing_values,
duplicates, outliers, validity, ...) hides *breadth* of problems. If
different rows each have one different kind of issue, every individual
detector reports a low percentage and the averaged score looks fine --
even though a large share of the actual rows in the dataset are
untrustworthy. That effect gets *worse*, not better, as row count
grows, since problems naturally spread across more distinct rows.

pct_problem_rows is a proportion, not a count, so it means the same
thing whether the dataset has 15 rows or 1,000,000: "X% of your rows
have at least one real, concrete issue." That makes it comparable
across datasets of any size without retuning thresholds per file.
"""

import re
import pandas as pd

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

DEFAULT_RANGE_RULES = {
    "age": (0, 120),
    "rating": (1, 5),
    "score": (0, 100),
    "percentage": (0, 100),
    "percent": (0, 100),
}

NON_NEGATIVE_NAME_HINTS = (
    "salary", "price", "cost", "amount", "revenue", "income",
    "quantity", "count", "fare", "wage", "pay",
)


def _find_columns_matching(df: pd.DataFrame, keyword: str) -> list:
    return [c for c in df.columns if keyword in str(c).strip().lower()]


def compute_row_quality(df: pd.DataFrame, numeric_cols: list,
                         id_like_cols: list = None, custom_range_rules: dict = None) -> dict:
    """
    Flag every row that has at least one concrete, individually-checkable
    problem, then score the dataset as the % of rows with NO flagged
    problems.

    Checks applied per row:
      - any missing value (excluding ID-like columns)
      - exact duplicate of another row (excluding ID-like columns)
      - malformed email in any *email* column
      - unparseable date in any *date* column
      - out-of-plausible-range numeric value (age/rating/salary/etc.)
      - statistical outlier (IQR) in any numeric column

    Returns
    -------
    dict with n_rows, n_problem_rows, pct_problem_rows, pct_clean_rows,
    a breakdown of how many rows each check flagged, and 'score' (==
    pct_clean_rows, 0-100, higher is better).
    """
    id_like_cols = id_like_cols or []
    n_rows = len(df)
    if n_rows == 0:
        return {"n_rows": 0, "n_problem_rows": 0, "pct_problem_rows": 0.0,
                "pct_clean_rows": 100.0, "score": 100.0, "breakdown": {}}

    problem_mask = pd.Series(False, index=df.index)
    breakdown = {}

    # 1. Missing values (any column, excluding IDs)
    check_cols = [c for c in df.columns if c not in id_like_cols]
    missing_mask = df[check_cols].isna().any(axis=1)
    breakdown["missing_values"] = int(missing_mask.sum())
    problem_mask |= missing_mask

    # 2. Exact duplicate rows (excluding IDs) -- keep=False flags ALL
    #    copies (originals included), since every copy is equally untrustworthy.
    dup_check_df = df.drop(columns=id_like_cols, errors="ignore")
    dup_mask = dup_check_df.duplicated(keep=False)
    breakdown["duplicates"] = int(dup_mask.sum())
    problem_mask |= dup_mask

    # 3. Malformed emails
    email_flagged = 0
    for col in _find_columns_matching(df, "email"):
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        invalid_idx = series[~series.str.match(EMAIL_REGEX)].index
        email_flagged += len(invalid_idx)
        problem_mask.loc[invalid_idx] = True
    breakdown["invalid_email"] = email_flagged

    # 4. Unparseable dates
    date_flagged = 0
    for col in _find_columns_matching(df, "date"):
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        invalid_idx = series[parsed.isna()].index
        date_flagged += len(invalid_idx)
        problem_mask.loc[invalid_idx] = True
    breakdown["invalid_date"] = date_flagged

    # 5. Out-of-plausible-range numeric values
    rules = dict(DEFAULT_RANGE_RULES)
    if custom_range_rules:
        rules.update(custom_range_rules)
    range_flagged = 0
    for col in numeric_cols:
        if col in id_like_cols:
            continue
        col_lower = str(col).strip().lower()
        bounds = next((rng for key, rng in rules.items() if key in col_lower), None)
        is_non_negative_only = bounds is None and any(hint in col_lower for hint in NON_NEGATIVE_NAME_HINTS)
        if bounds is None and not is_non_negative_only:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if bounds:
            lo, hi = bounds
            bad_idx = series[(series < lo) | (series > hi)].index
        else:
            bad_idx = series[series < 0].index
        range_flagged += len(bad_idx)
        problem_mask.loc[bad_idx] = True
    breakdown["out_of_range"] = range_flagged

    # 6. Statistical outliers (IQR), per numeric column
    outlier_flagged = 0
    for col in numeric_cols:
        if col in id_like_cols:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bad_idx = series[(series < lower) | (series > upper)].index
        outlier_flagged += len(bad_idx)
        problem_mask.loc[bad_idx] = True
    breakdown["statistical_outlier"] = outlier_flagged

    n_problem_rows = int(problem_mask.sum())
    pct_problem_rows = round((n_problem_rows / n_rows) * 100, 2)
    pct_clean_rows = round(100 - pct_problem_rows, 2)

    return {
        "n_rows": n_rows,
        "n_problem_rows": n_problem_rows,
        "pct_problem_rows": pct_problem_rows,
        "pct_clean_rows": pct_clean_rows,
        "breakdown": breakdown,
        "problem_row_indices": df.index[problem_mask].tolist()[:50],
        "score": pct_clean_rows,
    }