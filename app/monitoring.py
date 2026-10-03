"""
Production Telemetry & Drift Monitoring
=======================================
Tracks distribution shift (Kolmogorov-Smirnov Test, Population Stability Index)
and request latencies in real-time.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.stats import ks_2samp


class ModelDriftMonitor:
    """
    Monitors streaming ratings and predictions for data drift and popularity drift.
    """

    def __init__(self, baseline_ratings: np.ndarray):
        self.baseline_ratings = baseline_ratings
        self.latency_records: List[float] = []
        self.live_ratings: List[float] = []

    def record_latency(self, latency_ms: float):
        """Records inference latency in milliseconds."""
        self.latency_records.append(latency_ms)
        if len(self.latency_records) > 10000:
            self.latency_records.pop(0)

    def record_rating(self, rating: float):
        """Records incoming user feedback rating."""
        self.live_ratings.append(rating)

    def check_drift(self, new_ratings: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Runs two-sample Kolmogorov-Smirnov test between baseline and live/input ratings.
        """
        ratings_eval = np.array(new_ratings) if new_ratings is not None else np.array(self.live_ratings)
        if len(ratings_eval) < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "p_value": 1.0,
                "ks_statistic": 0.0,
                "drift_detected": False,
                "interpretation": "Insufficient data points for KS test."
            }

        stat, p_val = ks_2samp(self.baseline_ratings, ratings_eval)
        drift_detected = bool(p_val < 0.05)
        return {
            "status": "SUCCESS",
            "ks_statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": drift_detected,
            "sample_size": len(ratings_eval),
            "interpretation": "Significant distribution drift detected (p < 0.05)." if drift_detected else "No significant drift detected."
        }

    def check_rating_drift(self) -> Dict[str, Any]:
        return self.check_drift()

    def get_latency_summary(self) -> Dict[str, float]:
        """Calculates latency statistics."""
        if not self.latency_records:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        return {
            "p50": float(np.percentile(self.latency_records, 50)),
            "p95": float(np.percentile(self.latency_records, 95)),
            "p99": float(np.percentile(self.latency_records, 99)),
            "mean": float(np.mean(self.latency_records))
        }
