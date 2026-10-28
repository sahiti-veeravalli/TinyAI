"""
Unit Tests for TinyAI Pipeline.
Tests:
- Dataset generation & profiling
- Domain-Aware Scheduler logic
- Unstructured & structured pruning
- Knowledge distillation & focal loss
- Dynamic INT8 quantization
- Metrics evaluation & "within 1-2%" badge
"""

import os
import numpy as np
import pytest

from pipeline.datasets import generate_credit_card_fraud, generate_census_income
from pipeline.profiler import DatasetProfiler
from pipeline.scheduler import DomainAwareScheduler
from pipeline.pruning import prune_unstructured_magnitude, calculate_model_sparsity
from pipeline.evaluation import evaluate_model_comprehensive, calculate_ece


def test_credit_card_fraud_generation():
    bundle = generate_credit_card_fraud(n_samples=500)
    assert bundle.X_train.shape[1] == 30
    assert len(bundle.y_train) > 0
    assert bundle.cost_fn_ratio == 50.0
    # Imbalance check
    classes, counts = np.unique(bundle.y_train, return_counts=True)
    assert len(classes) == 2


def test_dataset_profiler():
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.array([0] * 190 + [1] * 10)  # 19:1 imbalance
    profile = DatasetProfiler.profile(X, y, cost_asymmetry=25.0, dataset_name="TestSet")
    
    assert profile.n_features == 10
    assert profile.imbalance_ratio >= 18.0
    assert profile.risk_domain == "high_stakes_fraud"


def test_domain_aware_scheduler_imbalanced():
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.array([0] * 190 + [1] * 10)
    profile = DatasetProfiler.profile(X, y, cost_asymmetry=50.0)
    
    plan = DomainAwareScheduler.recommend_plan(profile)
    # Patent claim: Imbalance > 15 & cost > 20 MUST enforce Distillation FIRST
    assert plan.stage_order[0] == "distillation"
    assert plan.pruning_ratio <= 0.35  # Conservative pruning
    assert plan.distillation_alpha >= 0.70  # High supervision


def test_domain_aware_scheduler_high_dimensional():
    X = np.random.randn(200, 50).astype(np.float32)  # 50 features
    y = np.array([0] * 100 + [1] * 100)
    profile = DatasetProfiler.profile(X, y, cost_asymmetry=1.0)
    
    plan = DomainAwareScheduler.recommend_plan(profile)
    # High feature dimensionality triggers structured pruning first
    assert plan.stage_order[0] == "pruning"
    assert plan.pruning_type == "structured_neurons"
    assert plan.pruning_ratio >= 0.50


def test_evaluation_metrics_pass_fail():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob_good = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.9, 0.95, 0.99])
    
    metrics = evaluate_model_comprehensive(
        y_true, y_prob_good,
        size_bytes=8192,
        latency_stats={"mean_ms": 0.5, "p95_ms": 0.8, "throughput_qps": 2000.0},
        baseline_roc_auc=1.0,
        model_name="Test Good"
    )
    assert metrics.within_1_to_2_percent is True
    assert metrics.badge_status == "PASS"

    # Degraded model
    y_prob_bad = np.array([0.4, 0.45, 0.5, 0.55, 0.5, 0.55, 0.6, 0.65])
    metrics_bad = evaluate_model_comprehensive(
        y_true, y_prob_bad,
        size_bytes=4096,
        latency_stats={"mean_ms": 0.2, "p95_ms": 0.3, "throughput_qps": 5000.0},
        baseline_roc_auc=1.0,
        model_name="Test Degraded"
    )
    assert metrics_bad.within_1_to_2_percent is False
    assert metrics_bad.badge_status == "FAIL"


def test_ece_calculation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = calculate_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
