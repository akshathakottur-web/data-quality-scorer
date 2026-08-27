"""
scoring.py

Combines detector findings into a single overall Data Quality Score.

Design note: the overall score is the ROW-LEVEL quality score from
row_quality.py -- the % of rows with zero flagged problems -- not a
weighted average of independent per-category percentages. Averaging
independent percentages hides breadth of problems and gets more
forgiving as row count grows (the same defect rate spreads across more
distinct rows, but each detector still only reports "a few percent").
The row-level score is a proportion, so it means the same thing at
15 rows or 15,000 rows.

The per-detector WEIGHTS below are still computed and returned --
they're useful as diagnostic sub-scores in the report/recommendations
(e.g. "your class imbalance is fine but your validity is poor") -- but
they no longer determine the headline number.
"""

WEIGHTS = {
    "missing_values": 0.132,
    "duplicates": 0.088,
    "outliers_iqr": 0.0704,
    "outliers_iforest": 0.088,
    "class_imbalance": 0.1056,
    "constant_columns": 0.0704,
    "correlated_features": 0.0616,
    "anomalies": 0.088,
    "label_issues": 0.088,
    "feature_leakage": 0.088,
    "validity": 0.12,
}


def compute_overall_score(all_reports: dict) -> dict:
    """
    Combine detector findings into individual sub-scores (diagnostic,
    weighted-average) AND the authoritative overall score (row-level,
    size-invariant, from all_reports['row_quality']).
    """
    individual_scores = {}
    active_weights = {}

    for key, weight in WEIGHTS.items():
        report = all_reports.get(key)
        if report is None:
            continue
        if report.get("applicable") is False:
            continue

        individual_scores[key] = report["score"]
        active_weights[key] = weight

    total_weight = sum(active_weights.values()) or 1.0
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    # Diagnostic-only: average of independent per-category scores.
    diagnostic_avg_score = sum(
        individual_scores[k] * normalized_weights[k] for k in individual_scores
    )
    diagnostic_avg_score = round(diagnostic_avg_score, 1)

    # Authoritative: row-level, size-invariant score.
    row_quality = all_reports.get("row_quality")
    if row_quality is not None:
        overall_score = row_quality["score"]
    else:
        # Fallback if row_quality wasn't run for some reason.
        overall_score = diagnostic_avg_score

    status = get_status_label(overall_score)

    return {
        "individual_scores": individual_scores,
        "weights_used": normalized_weights,
        "diagnostic_avg_score": diagnostic_avg_score,
        "overall_score": overall_score,
        "status": status,
    }


def get_status_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    elif score >= 40:
        return "Poor"
    else:
        return "Critical"