"""
recommendations.py

Generates human-readable, actionable recommendations based on
detector findings.
"""


def generate_recommendations(all_reports: dict) -> list:
    """
    Build a list of plain-English recommendations based on which
    issues were detected and how severe they are.
    """
    recs = []

    mv = all_reports.get("missing_values", {})
    if mv.get("high_missing_columns"):
        cols = ", ".join(mv["high_missing_columns"])
        recs.append(
            f"Columns [{cols}] have over 40% missing values. "
            f"Consider dropping them or using advanced imputation (e.g. KNN imputation)."
        )
    elif mv.get("pct_missing_overall", 0) > 5:
        recs.append(
            "Moderate missing data detected. Consider imputing with mean/median "
            "(numeric) or mode (categorical), or using a model-based imputer."
        )

    dup = all_reports.get("duplicates", {})
    if dup.get("n_duplicate_rows", 0) > 0:
        recs.append(
            f"Found {dup['n_duplicate_rows']} duplicate rows. "
            f"Consider removing them with df.drop_duplicates() unless duplication is intentional."
        )

    validity = all_reports.get("validity", {})
    if validity.get("applicable"):
        email_bad = {
            col: v for col, v in validity.get("email_validity", {}).items() if v.get("n_invalid", 0) > 0
        }
        if email_bad:
            details = ", ".join(f"{col} ({v['n_invalid']} invalid)" for col, v in email_bad.items())
            recs.append(
                f"Malformed email addresses found in: {details}. "
                f"Validate against a proper email regex or a verification service before using these records."
            )

        date_bad = {
            col: v for col, v in validity.get("date_validity", {}).items() if v.get("n_invalid", 0) > 0
        }
        if date_bad:
            details = ", ".join(f"{col} ({v['n_invalid']} invalid)" for col, v in date_bad.items())
            recs.append(
                f"Unparseable or invalid dates found in: {details}. "
                f"Standardize on a single date format (e.g. ISO 8601 YYYY-MM-DD) at the data-entry or ingestion layer."
            )

        range_bad = {
            col: v for col, v in validity.get("range_validity", {}).items() if v.get("n_invalid", 0) > 0
        }
        if range_bad:
            details = ", ".join(f"{col} ({v['n_invalid']} out of plausible range)" for col, v in range_bad.items())
            recs.append(
                f"Out-of-range values found in: {details}. "
                f"These are implausible regardless of statistical distribution (e.g. negative salary, "
                f"age over 120, rating above scale max) -- review and correct or drop these rows."
            )

    iqr = all_reports.get("outliers_iqr", {})
    if iqr.get("pct_overall", 0) > 5:
        recs.append(
            "Significant statistical outliers detected. Investigate whether these are "
            "data entry errors or legitimate rare events before deciding to cap, transform, or remove them."
        )

    iforest = all_reports.get("outliers_iforest", {})
    if iforest.get("pct_outliers", 0) > 5:
        recs.append(
            "Isolation Forest flagged multivariate outliers — combinations of feature values "
            "that are individually normal but jointly unusual. Review the flagged rows manually."
        )

    imb = all_reports.get("class_imbalance", {})
    if imb.get("applicable") and imb.get("imbalance_ratio", 1) > 3:
        recs.append(
            f"Class imbalance detected (ratio {imb['imbalance_ratio']}:1). Consider resampling "
            f"techniques (SMOTE, undersampling) or class-weighted loss functions during training."
        )

    const = all_reports.get("constant_columns", {})
    if const.get("constant_columns"):
        cols = ", ".join(const["constant_columns"])
        recs.append(f"Columns [{cols}] have zero variance and provide no signal — safe to drop.")

    corr = all_reports.get("correlated_features", {})
    if corr.get("correlated_pairs"):
        pairs = ", ".join(f"{p['column_1']}~{p['column_2']}" for p in corr["correlated_pairs"])
        recs.append(f"Highly correlated feature pairs found ({pairs}). Consider dropping one from each pair.")

    anom = all_reports.get("anomalies", {})
    if anom.get("n_anomalies", 0) > 0:
        recs.append(
            f"{anom['n_anomalies']} records don't fit any typical cluster pattern in the data. "
            f"Review these manually — they may be errors or genuinely rare cases worth understanding."
        )

    label = all_reports.get("label_issues", {})
    if label.get("applicable") and label.get("n_suspicious_labels", 0) > 0:
        recs.append(
            f"{label['n_suspicious_labels']} rows have labels the model strongly disagrees with. "
            f"These may be mislabeled — manual review recommended before training."
        )

    leak = all_reports.get("feature_leakage", {})
    if leak.get("applicable") and leak.get("suspicious_features"):
        feats = ", ".join(f["feature"] for f in leak["suspicious_features"])
        recs.append(
            f"Feature(s) [{feats}] predict the target suspiciously well alone. "
            f"Verify these aren't leaking future information not available at prediction time."
        )

    if not recs:
        recs.append("No major data quality issues detected. Dataset looks ready for modeling.")

    return recs