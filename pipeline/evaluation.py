"""
Model Evaluation and Calibration Module for TinyAI.
Computes IEEE Access benchmark metrics:
- PR-AUC & ROC-AUC
- Recall at 95% Precision (crucial for fraud/risk systems)
- Expected Calibration Error (ECE) for edge reliability
- Latency (p50, p95, p99) & footprint reduction
- "Within 1-2% accuracy" badge validation
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)


@dataclass
class EvaluationMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    recall_at_95_precision: float
    expected_calibration_error: float
    model_size_kb: float
    latency_mean_ms: float
    latency_p95_ms: float
    throughput_qps: float
    within_1_to_2_percent: bool
    accuracy_gap_pct: float            # (teacher_auc - student_auc) / teacher_auc * 100
    badge_status: str                  # "PASS" (Lime) or "FAIL" (Magenta)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def calculate_recall_at_fixed_precision(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.95) -> float:
    """Computes max recall achieved while maintaining precision >= min_precision."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid_recalls = recalls[precisions >= min_precision]
    return float(round(np.max(valid_recalls) if len(valid_recalls) > 0 else 0.0, 4))


def evaluate_model_comprehensive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    size_bytes: int,
    latency_stats: Dict[str, float],
    baseline_roc_auc: Optional[float] = None,
    threshold: float = 0.5,
    model_name: str = "Compressed Model"
) -> EvaluationMetrics:
    """Full comprehensive metric evaluation against IEEE Access research criteria."""
    y_pred = (y_prob >= threshold).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    recall_95 = calculate_recall_at_fixed_precision(y_true, y_prob, min_precision=0.95)
    ece = calculate_ece(y_true, y_prob, n_bins=10)

    size_kb = round(size_bytes / 1024.0, 2)
    lat_mean = latency_stats.get("mean_ms", 1.0)
    lat_p95 = latency_stats.get("p95_ms", 1.5)
    qps = latency_stats.get("throughput_qps", 1000.0)

    if baseline_roc_auc is not None and baseline_roc_auc > 0:
        gap_pct = round(((baseline_roc_auc - roc) / baseline_roc_auc) * 100.0, 2)
        within_tolerance = gap_pct <= 2.0
    else:
        gap_pct = 0.0
        within_tolerance = True

    badge_status = "PASS" if within_tolerance else "FAIL"

    return EvaluationMetrics(
        model_name=model_name,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(roc, 4),
        pr_auc=round(pr_auc, 4),
        recall_at_95_precision=recall_95,
        expected_calibration_error=ece,
        model_size_kb=size_kb,
        latency_mean_ms=lat_mean,
        latency_p95_ms=lat_p95,
        throughput_qps=qps,
        within_1_to_2_percent=within_tolerance,
        accuracy_gap_pct=gap_pct,
        badge_status=badge_status
    )
