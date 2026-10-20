from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculates comprehensive classification metrics suited for both balanced and imbalanced tasks."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "weighted", zero_division=0)),
    }

    if y_prob is not None:
        try:
            # If 2D probability matrix for binary classification, take positive class prob
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                pos_prob = y_prob[:, 1]
            else:
                pos_prob = y_prob

            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_prob))
                metrics["pr_auc"] = float(average_precision_score(y_true, pos_prob))
                metrics["brier_score"] = float(brier_score_loss(y_true, pos_prob))
                metrics["log_loss"] = float(log_loss(y_true, pos_prob, labels=[0, 1]))
        except Exception:
            # Fallback if probability metric cannot be computed (e.g. single class in split)
            pass

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculates standard regression error and goodness-of-fit metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred))
    }
