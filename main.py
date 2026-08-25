"""
main.py

Full pipeline: load -> profile -> run all detectors -> score -> report.
"""

from src.dqs.profiler import load_csv, profile_dataset, print_profile_summary
from src.dqs.missing_duplicates import analyze_missing_values, analyze_duplicates
from src.dqs.outliers import detect_outliers_iqr, detect_outliers_isolation_forest
from src.dqs.imbalance import analyze_class_imbalance
from src.dqs.redundant_features import detect_constant_columns, detect_correlated_features
from src.dqs.anomalies import detect_anomalous_records
from src.dqs.label_leakage import detect_label_issues, detect_feature_leakage
from src.dqs.scoring import compute_overall_score
from src.dqs.recommendations import generate_recommendations
from src.dqs.report import generate_text_report, save_text_report, generate_visualizations


def run_pipeline(filepath: str) -> dict:
    df = load_csv(filepath)
    profile = profile_dataset(df)
    target = profile["guessed_target"]
    numeric_cols = profile["numeric_cols"]

    all_reports = {
        "missing_values": analyze_missing_values(df),
        "duplicates": analyze_duplicates(df),
        "outliers_iqr": detect_outliers_iqr(df, numeric_cols),
        "outliers_iforest": detect_outliers_isolation_forest(df, numeric_cols),
        "class_imbalance": analyze_class_imbalance(df, target),
        "constant_columns": detect_constant_columns(df),
        "correlated_features": detect_correlated_features(df, numeric_cols),
        "anomalies": detect_anomalous_records(df, numeric_cols),
        "label_issues": detect_label_issues(df, target, numeric_cols),
        "feature_leakage": detect_feature_leakage(df, target, numeric_cols),
    }

    scoring = compute_overall_score(all_reports)
    recommendations = generate_recommendations(all_reports)

    report_text = generate_text_report(profile, all_reports, scoring, recommendations)
    save_text_report(report_text)

    all_reports["_individual_scores"] = scoring["individual_scores"]
    generate_visualizations(df, profile, all_reports)

    return {
        "profile": profile,
        "all_reports": all_reports,
        "scoring": scoring,
        "recommendations": recommendations,
        "report_text": report_text,
    }


def main():
    filepath = "data/synthetic_bad.csv"
    print(f"Running Data Quality Scoring pipeline on: {filepath}\n")

    results = run_pipeline(filepath)

    print(results["report_text"])


if __name__ == "__main__":
    main()