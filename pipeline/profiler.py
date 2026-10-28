"""
Dataset Profiler module for TinyAI.
Extracts structural and cost characteristics of tabular datasets to drive
the Domain-Aware Compression Scheduler.
Patentable aspect: Coupling dataset topological and economic metrics directly
to the ordering and intensity of pruning, distillation, and quantization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    dataset_name: str
    n_samples: int
    n_features: int
    n_numerical: int
    n_categorical: int
    categorical_ratio: float
    imbalance_ratio: float      # Majority : Minority ratio (e.g. 50.0 = 50:1)
    minority_class_rate: float  # e.g. 0.019 (1.9%)
    missing_value_rate: float   # 0.0 to 1.0
    cost_asymmetry: float       # Cost(FN) / Cost(FP), e.g. 50.0 for fraud
    sparsity_rate: float        # Fraction of exact zeros
    dominant_correlation: float # Max pairwise feature correlation
    feature_names: List[str]
    risk_domain: str            # 'high_stakes_fraud', 'credit_risk', 'general_classification'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetProfiler:
    """Profiles tabular data arrays or dataframes to derive domain compression heuristics."""

    @staticmethod
    def profile(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_indices: Optional[List[int]] = None,
        cost_asymmetry: float = 1.0,
        dataset_name: str = "Custom Dataset"
    ) -> DatasetProfile:
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]
            
        cat_idx = categorical_indices or []
        n_categorical = len(cat_idx)
        n_numerical = n_features - n_categorical
        categorical_ratio = float(n_categorical / max(n_features, 1))

        # Class imbalance calculation
        classes, counts = np.unique(y, return_counts=True)
        if len(counts) >= 2:
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_ratio = float(max_count / max(min_count, 1))
            minority_rate = float(min_count / n_samples)
        else:
            imbalance_ratio = 1.0
            minority_rate = 0.5

        # Missing values (NaN / Inf)
        missing_count = np.isnan(X).sum() + np.isinf(X).sum()
        missing_value_rate = float(missing_count / max(X.size, 1))

        # Sparsity (exact zeros)
        sparsity_rate = float((X == 0.0).sum() / max(X.size, 1))

        # Correlation analysis (subsampled for speed on large matrices)
        sub_samples = min(n_samples, 2000)
        sub_X = np.nan_to_num(X[:sub_samples], nan=0.0)
        corr_matrix = np.corrcoef(sub_X, rowvar=False)
        np.fill_diagonal(corr_matrix, 0.0)
        dominant_correlation = float(np.max(np.abs(np.nan_to_num(corr_matrix, nan=0.0))))

        # Identify risk domain
        if imbalance_ratio > 20.0 or cost_asymmetry >= 25.0:
            risk_domain = "high_stakes_fraud"
        elif imbalance_ratio > 4.0 or cost_asymmetry >= 3.0:
            risk_domain = "credit_risk"
        else:
            risk_domain = "general_classification"

        return DatasetProfile(
            dataset_name=dataset_name,
            n_samples=int(n_samples),
            n_features=int(n_features),
            n_numerical=int(n_numerical),
            n_categorical=int(n_categorical),
            categorical_ratio=round(categorical_ratio, 4),
            imbalance_ratio=round(imbalance_ratio, 2),
            minority_class_rate=round(minority_rate, 4),
            missing_value_rate=round(missing_value_rate, 4),
            cost_asymmetry=round(cost_asymmetry, 2),
            sparsity_rate=round(sparsity_rate, 4),
            dominant_correlation=round(dominant_correlation, 4),
            feature_names=feature_names,
            risk_domain=risk_domain
        )
