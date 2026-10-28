"""
Edge Drift Monitoring module for TinyAI.
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests on
deployed edge tabular data streams to detect distribution shifts and trigger
automated re-compression scheduling.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and production edge stream.
    PSI < 0.10: Stable (Green)
    0.10 <= PSI <= 0.25: Moderate drift (Amber)
    PSI > 0.25: Critical drift (Magenta) -> Trigger automated recompression
    """
    # Exclude non-finite
    b = baseline[np.isfinite(baseline)]
    c = current[np.isfinite(current)]
    if len(b) < 10 or len(c) < 10:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(b, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    b_counts, _ = np.histogram(b, bins=bin_edges)
    c_counts, _ = np.histogram(c, bins=bin_edges)

    b_pct = np.maximum(b_counts / len(b), 1e-4)
    c_pct = np.maximum(c_counts / len(c), 1e-4)

    psi_val = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(round(psi_val, 4))


def simulate_stream_drift(
    baseline_X: np.ndarray,
    n_stream: int = 500,
    drift_level: float = 0.35,
    drifted_features: List[int] = [0, 1, 2]
) -> np.ndarray:
    """Simulates an edge production stream with covariate drift."""
    rng = np.random.RandomState(int(drift_level * 1000))
    sample_idx = rng.choice(len(baseline_X), size=n_stream, replace=True)
    stream_X = baseline_X[sample_idx].copy()

    for feat_idx in drifted_features:
        if feat_idx < stream_X.shape[1]:
            # Shift mean and inflate variance
            shift = drift_level * np.std(stream_X[:, feat_idx]) * 2.5
            stream_X[:, feat_idx] += shift + rng.normal(0, drift_level, size=n_stream)

    return stream_X


def evaluate_stream_drift(
    baseline_X: np.ndarray,
    current_stream_X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Evaluates multi-feature drift across production stream."""
    feature_drifts = []
    max_psi = 0.0

    for i in range(min(baseline_X.shape[1], 15)):
        feat_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
        psi = calculate_psi(baseline_X[:, i], current_stream_X[:, i])
        ks_stat, p_val = ks_2samp(baseline_X[:, i], current_stream_X[:, i])
        
        status = "CRITICAL" if psi > 0.25 else ("MODERATE" if psi >= 0.10 else "STABLE")
        if psi > max_psi:
            max_psi = psi

        feature_drifts.append({
            "feature": feat_name,
            "psi": psi,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 6),
            "status": status
        })

    # Sort by PSI descending
    feature_drifts.sort(key=lambda x: x["psi"], reverse=True)

    recompression_required = max_psi > 0.25

    return {
        "max_psi": round(max_psi, 4),
        "overall_status": "CRITICAL DRIFT" if recompression_required else ("MODERATE DRIFT" if max_psi >= 0.10 else "STABLE"),
        "recompression_required": recompression_required,
        "recommendation": (
            "Distribution shift detected across key predictive inputs. Re-run Domain-Aware Scheduler "
            "to adapt pruning thresholds and re-calibrate student distillation."
            if recompression_required else "Edge pipeline within acceptable operating tolerances."
        ),
        "features": feature_drifts
    }
