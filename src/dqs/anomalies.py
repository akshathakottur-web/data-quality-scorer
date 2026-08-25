"""
anomalies.py

Detects anomalous records using K-Means clustering distance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def detect_anomalous_records(df: pd.DataFrame, numeric_cols: list,
                              n_clusters: int = 5,
                              anomaly_percentile: float = 95.0) -> dict:
    """
    Cluster rows using K-Means, then flag rows whose distance to their
    nearest cluster center is unusually large (top percentile).

    Returns
    -------
    dict with anomalous row indices, count, and score.
    """
    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    if numeric_df.shape[1] == 0 or numeric_df.shape[0] < n_clusters * 2:
        return {
            "n_anomalies": 0,
            "pct_anomalies": 0.0,
            "anomalous_indices": [],
            "score": 100.0,
            "note": "Not enough numeric data to run clustering.",
        }

    # Scale features so columns with large ranges (e.g. Fare: 0-500)
    # don't dominate distance calculations over small-range columns (e.g. Pclass: 1-3)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    k = min(n_clusters, max(2, numeric_df.shape[0] // 10))
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = model.fit_predict(scaled)

    # Distance from each point to its assigned cluster's center
    distances = np.linalg.norm(scaled - model.cluster_centers_[cluster_labels], axis=1)

    threshold = np.percentile(distances, anomaly_percentile)
    anomaly_mask = distances > threshold

    n_anomalies = int(anomaly_mask.sum())
    pct_anomalies = round((n_anomalies / len(df)) * 100, 2)
    anomalous_indices = df.index[anomaly_mask].tolist()[:20]

    score = 100 - min(pct_anomalies * 3, 100)
    score = max(0, round(score, 1))

    return {
        "n_clusters_used": k,
        "n_anomalies": n_anomalies,
        "pct_anomalies": pct_anomalies,
        "anomalous_indices": anomalous_indices,
        "score": score,
    }