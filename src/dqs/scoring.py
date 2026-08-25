"""
scoring.py

Combines all individual detector scores into a single overall
Data Quality Score using a weighted average.
"""

# Weights reflect relative importance to ML model quality.
# These sum to 1.0 and can be tuned.
WEIGHTS = {
    "missing_values": 0.15,
    "duplicates": 0.10,
    "outliers_iqr": 0.08,
    "outliers_iforest": 0.10,
    "class_imbalance": 0.12,
    "constant_columns": 0.08,
    "correlated_features": 0.07,
    "anomalies": 0.10,
    "label_issues": 0.10,
    "feature_leakage": 0.10,
}


def compute_overall_score(all_reports: dict) -> dict:
    """
    Combine individual detector scores into one overall score.

    Parameters
    ----------
    all_reports : dict
        Dictionary mapping detector name -> its result dict
        (must contain a 'score' key, or 'applicable': False to be skipped
        and re-normalized around).

    Returns
    -------
    dict with individual_scores, overall_score, and status label.
    """
    individual_scores = {}
    active_weights = {}

    for key, weight in WEIGHTS.items():
        report = all_reports.get(key)
        if report is None:
            continue
        if report.get("applicable") is False:
            continue  # skip detectors that didn't apply (e.g. no target column)

        individual_scores[key] = report["score"]
        active_weights[key] = weight

    # Re-normalize weights so they sum to 1.0 even if some detectors were skipped
    total_weight = sum(active_weights.values()) or 1.0
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    overall_score = sum(
        individual_scores[k] * normalized_weights[k] for k in individual_scores
    )
    overall_score = round(overall_score, 1)

    status = get_status_label(overall_score)

    return {
        "individual_scores": individual_scores,
        "weights_used": normalized_weights,
        "overall_score": overall_score,
        "status": status,
    }


def get_status_label(score: float) -> str:
    """Map a numeric score to a human-readable status."""
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