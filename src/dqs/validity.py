"""
validity.py

Detects format and domain-validity issues that pure statistics miss:
malformed emails, unparseable dates, and out-of-plausible-range values
(e.g. age 150, a 1-5 rating of 5.5, a negative salary).

This complements outliers.py: IQR/Isolation Forest catch values that
are statistically unusual *relative to the rest of the column*. This
file catches values that are wrong *by definition*, regardless of
what the rest of the column looks like -- a malformed email or an
invalid calendar date is wrong even in a dataset full of other
malformed emails and invalid dates.
"""

import re
import pandas as pd

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Built-in sensible bounds for common column-name patterns. Matched by
# substring against the (lowercased) column name -- "age" matches
# "age", "customer_age", etc.
DEFAULT_RANGE_RULES = {
    "age": (0, 120),
    "rating": (1, 5),
    "score": (0, 100),
    "percentage": (0, 100),
    "percent": (0, 100),
}

# Columns whose name suggests the value should never be negative, used
# when no explicit range rule matches (e.g. "salary" has no fixed
# upper bound, but it should never be negative).
NON_NEGATIVE_NAME_HINTS = (
    "salary", "price", "cost", "amount", "revenue", "income",
    "quantity", "count", "fare", "wage", "pay",
)


def _find_columns_matching(df: pd.DataFrame, keyword: str) -> list:
    return [c for c in df.columns if keyword in str(c).strip().lower()]


def _validate_emails(df: pd.DataFrame) -> dict:
    results = {}
    total_invalid = 0
    total_checked = 0

    for col in _find_columns_matching(df, "email"):
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        valid_mask = series.str.match(EMAIL_REGEX)
        n_invalid = int((~valid_mask).sum())

        results[col] = {
            "n_checked": int(len(series)),
            "n_invalid": n_invalid,
            "pct_invalid": round(n_invalid / len(series) * 100, 2),
            "invalid_examples": series[~valid_mask].tolist()[:5],
        }
        total_invalid += n_invalid
        total_checked += len(series)

    return results, total_invalid, total_checked


def _validate_dates(df: pd.DataFrame) -> dict:
    results = {}
    total_invalid = 0
    total_checked = 0

    for col in _find_columns_matching(df, "date"):
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        n_invalid = int(parsed.isna().sum())

        results[col] = {
            "n_checked": int(len(series)),
            "n_invalid": n_invalid,
            "pct_invalid": round(n_invalid / len(series) * 100, 2),
            "invalid_examples": series[parsed.isna()].tolist()[:5],
        }
        total_invalid += n_invalid
        total_checked += len(series)

    return results, total_invalid, total_checked


def _validate_ranges(df: pd.DataFrame, numeric_cols: list,
                      id_like_cols: list = None, custom_rules: dict = None) -> dict:
    id_like_cols = id_like_cols or []
    rules = dict(DEFAULT_RANGE_RULES)
    if custom_rules:
        rules.update(custom_rules)

    results = {}
    total_invalid = 0
    total_checked = 0

    for col in numeric_cols:
        if col in id_like_cols:
            continue

        col_lower = str(col).strip().lower()
        bounds = next((rng for key, rng in rules.items() if key in col_lower), None)
        is_non_negative_only = bounds is None and any(hint in col_lower for hint in NON_NEGATIVE_NAME_HINTS)

        if bounds is None and not is_non_negative_only:
            continue  # No applicable rule for this column -- nothing to check.

        series = df[col].dropna()
        if series.empty:
            continue

        if bounds:
            lo, hi = bounds
            invalid_mask = (series < lo) | (series > hi)
        else:
            invalid_mask = series < 0

        n_invalid = int(invalid_mask.sum())

        results[col] = {
            "bounds_used": bounds if bounds else "non-negative",
            "n_checked": int(len(series)),
            "n_invalid": n_invalid,
            "pct_invalid": round(n_invalid / len(series) * 100, 2),
            "invalid_examples": series[invalid_mask].tolist()[:5],
        }
        total_invalid += n_invalid
        total_checked += len(series)

    return results, total_invalid, total_checked


def detect_data_validity(df: pd.DataFrame, numeric_cols: list = None,
                          id_like_cols: list = None, custom_range_rules: dict = None) -> dict:
    """
    Run all validity checks (email format, date parseability, plausible
    numeric ranges) and combine them into one score.

    Parameters
    ----------
    numeric_cols : list
        From profile['numeric_cols'] (already excludes ID-like columns).
    id_like_cols : list
        From profile['id_like_cols'] -- passed through so range checks
        never accidentally flag an ID column.
    custom_range_rules : dict, optional
        Override or extend DEFAULT_RANGE_RULES, e.g. {"rating": (1, 10)}
        if your rating scale isn't 1-5.

    Returns
    -------
    dict with per-check breakdowns and an overall score (0-100).
    """
    numeric_cols = numeric_cols or []

    email_results, email_invalid, email_checked = _validate_emails(df)
    date_results, date_invalid, date_checked = _validate_dates(df)
    range_results, range_invalid, range_checked = _validate_ranges(
        df, numeric_cols, id_like_cols, custom_range_rules
    )

    total_invalid = email_invalid + date_invalid + range_invalid
    total_checked = email_checked + date_checked + range_checked

    if total_checked == 0:
        return {
            "applicable": False,
            "reason": "No email/date/range-checkable columns found.",
            "score": 100.0,
        }

    pct_invalid = round((total_invalid / total_checked) * 100, 2)
    score = 100 - min(pct_invalid * 5, 100)
    score = max(0, round(score, 1))

    return {
        "applicable": True,
        "email_validity": email_results,
        "date_validity": date_results,
        "range_validity": range_results,
        "total_invalid": total_invalid,
        "total_checked": total_checked,
        "pct_invalid": pct_invalid,
        "score": score,
    }