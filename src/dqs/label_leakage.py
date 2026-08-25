"""
label_leakage.py

Detects potential label issues (via classifier disagreement) and
feature leakage (via suspiciously high single-feature predictive power).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


def detect_label_issues(df: pd.DataFrame, target_col: str | None,
                         numeric_cols: list, confidence_threshold: float = 0.9) -> dict:
    """
    Train a simple classifier and flag rows where the model strongly
    disagrees with the given label (potential mislabeling).
    """
    if target_col is None or target_col not in df.columns:
        return {"applicable": False, "reason": "No target column.", "score": 100.0}

    feature_cols = [c for c in numeric_cols if c != target_col]
    if len(feature_cols) < 2:
        return {"applicable": False, "reason": "Not enough numeric features.", "score": 100.0}

    work_df = df[feature_cols + [target_col]].dropna()
    if len(work_df) < 30:
        return {"applicable": False, "reason": "Not enough rows after dropping missing values.", "score": 100.0}

    X = work_df[feature_cols]
    y_raw = work_df[target_col]

    n_unique = y_raw.nunique()
    if n_unique > 20 or n_unique / len(y_raw) > 0.5:
        return {"applicable": False, "reason": "Target looks continuous, not classification.", "score": 100.0}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)

    # cross_val_predict trains on folds the row was NOT in, so predictions
    # are honest (not the model just memorizing the row it was trained on)
    try:
        probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")
    except Exception as e:
        return {"applicable": False, "reason": f"Could not run classifier: {e}", "score": 100.0}

    predicted_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    # Suspicious: model is confident AND disagrees with the given label
    suspicious_mask = (predicted_class != y) & (confidence >= confidence_threshold)
    n_suspicious = int(suspicious_mask.sum())
    pct_suspicious = round((n_suspicious / len(work_df)) * 100, 2)

    suspicious_indices = work_df.index[suspicious_mask].tolist()[:20]

    score = 100 - min(pct_suspicious * 5, 100)
    score = max(0, round(score, 1))

    return {
        "applicable": True,
        "n_suspicious_labels": n_suspicious,
        "pct_suspicious": pct_suspicious,
        "suspicious_indices": suspicious_indices,
        "score": score,
    }


def detect_feature_leakage(df: pd.DataFrame, target_col: str | None,
                            numeric_cols: list, leakage_threshold: float = 0.95) -> dict:
    """
    Check each feature individually: if a single feature alone predicts
    the target with suspiciously high accuracy, flag it as possible leakage.
    """
    if target_col is None or target_col not in df.columns:
        return {"applicable": False, "reason": "No target column.", "score": 100.0}

    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        return {"applicable": False, "reason": "No numeric features to check.", "score": 100.0}

    work_df = df[feature_cols + [target_col]].dropna()
    if len(work_df) < 30:
        return {"applicable": False, "reason": "Not enough rows.", "score": 100.0}

    y_raw = work_df[target_col]
    n_unique = y_raw.nunique()
    if n_unique > 20 or n_unique / len(y_raw) > 0.5:
        return {"applicable": False, "reason": "Target looks continuous.", "score": 100.0}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    suspicious_features = []
    for col in feature_cols:
        X_single = work_df[[col]]
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=4)
        try:
            preds = cross_val_predict(model, X_single, y, cv=5)
        except Exception:
            continue

        accuracy = (preds == y).mean()
        if accuracy >= leakage_threshold:
            suspicious_features.append({
                "feature": col,
                "single_feature_accuracy": round(float(accuracy), 4),
            })

    score = 100 - min(len(suspicious_features) * 25, 100)
    score = max(0, round(score, 1))

    return {
        "applicable": True,
        "suspicious_features": suspicious_features,
        "score": score,
    }