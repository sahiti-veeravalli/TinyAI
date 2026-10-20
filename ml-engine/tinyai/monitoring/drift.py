from typing import Dict, Any, List, Optional
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, Field


class FeatureDriftReport(BaseModel):
    feature_name: str
    ks_statistic: float
    p_value: float
    psi_score: float
    drift_detected: bool
    drift_severity: str  # "NONE", "LOW", "MODERATE", "CRITICAL"
    reference_mean: float
    current_mean: float


class DatasetDriftAnalysis(BaseModel):
    overall_drift_detected: bool
    drifted_features_count: int
    total_features_count: int
    feature_reports: List[FeatureDriftReport]
    recommendation: str


class DriftMonitor:
    """Statistical data drift detector calculating two-sample Kolmogorov-Smirnov test and Population Stability Index (PSI)."""

    @classmethod
    def calculate_psi(cls, expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index between reference baseline and production inference features."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        # Drop duplicates in bin edges
        bins = np.percentile(expected, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        # Clip edges for safety
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # Smooth zero counts
        eps = 1e-4
        expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
        actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi_val))

    @classmethod
    def analyze_drift(
        cls,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
        ks_alpha: float = 0.01,
        psi_threshold: float = 0.20
    ) -> DatasetDriftAnalysis:
        reports: List[FeatureDriftReport] = []
        drifted_count = 0

        n_features = min(reference_data.shape[1], current_data.shape[1], len(feature_names))

        for i in range(n_features):
            feat_name = feature_names[i]
            ref_col = reference_data[:, i]
            curr_col = current_data[:, i]

            # 2-sample Kolmogorov-Smirnov test
            ks_res = stats.ks_2samp(ref_col, curr_col)
            stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            # PSI
            psi_score = cls.calculate_psi(ref_col, curr_col)

            # Determine drift severity
            is_drift = (p_val < ks_alpha) or (psi_score >= psi_threshold)
            if is_drift:
                drifted_count += 1
                if psi_score > 0.4 or p_val < 1e-6:
                    severity = "CRITICAL"
                elif psi_score > 0.2 or p_val < 1e-3:
                    severity = "MODERATE"
                else:
                    severity = "LOW"
            else:
                severity = "NONE"

            reports.append(FeatureDriftReport(
                feature_name=feat_name,
                ks_statistic=round(stat, 4),
                p_value=round(p_val, 6),
                psi_score=round(psi_score, 4),
                drift_detected=is_drift,
                drift_severity=severity,
                reference_mean=round(float(np.mean(ref_col)), 3),
                current_mean=round(float(np.mean(curr_col)), 3)
            ))

        overall_drift = (drifted_count / max(1, n_features)) >= 0.25

        if overall_drift:
            rec = f"Alert: {drifted_count}/{n_features} features display significant distribution divergence. Model retraining or edge calibration recommended."
        else:
            rec = "Distributions remain stable within statistical confidence intervals."

        return DatasetDriftAnalysis(
            overall_drift_detected=overall_drift,
            drifted_features_count=drifted_count,
            total_features_count=n_features,
            feature_reports=reports,
            recommendation=rec
        )
