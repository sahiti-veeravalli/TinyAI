"""
Pydantic v2 schemas and data validation models for TinyAI API.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


# Profile
class DatasetProfileRequest(BaseModel):
    dataset_name: str = Field("Credit Card Fraud", description="Benchmark dataset ID or custom name")
    custom_samples: Optional[int] = Field(3000, description="Sample size cap")
    cost_asymmetry: Optional[float] = Field(50.0, description="Cost ratio: False Negative / False Positive")


class DatasetProfileResponse(BaseModel):
    dataset_name: str
    n_samples: int
    n_features: int
    n_numerical: int
    n_categorical: int
    categorical_ratio: float
    imbalance_ratio: float
    minority_class_rate: float
    missing_value_rate: float
    cost_asymmetry: float
    sparsity_rate: float
    dominant_correlation: float
    feature_names: List[str]
    risk_domain: str


# Scheduler
class RecommendPlanRequest(BaseModel):
    profile: DatasetProfileResponse


class CompressionPlanResponse(BaseModel):
    dataset_name: str
    stage_order: List[str]
    pruning_type: str
    pruning_ratio: float
    distillation_temperature: float
    distillation_alpha: float
    focal_gamma: float
    quantization_bits: int
    quantization_type: str
    rationale: List[str]
    expected_size_reduction_x: float
    expected_latency_speedup_x: float
    expected_accuracy_retention_pct: float


class OptunaSearchRequest(BaseModel):
    dataset_name: str = "Credit Card Fraud"
    cost_asymmetry: float = 50.0
    n_trials: int = 15


class ParetoPoint(BaseModel):
    trial_id: int
    stage_order: List[str]
    pruning_ratio: float
    distillation_temperature: float
    distillation_alpha: float
    quantization_bits: int
    roc_auc: float
    pr_auc: float
    size_kb: float
    latency_ms: float
    is_pareto_optimal: bool
    is_domain_aware: bool
    status: str


# Compression
class CompressJobRequest(BaseModel):
    dataset_name: str = "Credit Card Fraud"
    stage_order: Optional[List[str]] = None
    pruning_ratio: float = 0.30
    distillation_temperature: float = 3.5
    distillation_alpha: float = 0.75
    quantization_bits: int = 8


# Prediction & Inference
class DualPredictRequest(BaseModel):
    dataset_name: str = "credit_card_fraud"
    features: List[float]


class DualPredictResponse(BaseModel):
    teacher_probability: float
    student_probability: float
    teacher_prediction: int
    student_prediction: int
    agreement: bool
    teacher_latency_ms: float
    student_latency_ms: float
    speedup_ratio: float
    confidence_delta: float


# Explainability & Fairness
class ExplainRequest(BaseModel):
    dataset_name: str = "credit_card_fraud"
    n_features: int = 10


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    std: float


class ExplainResponse(BaseModel):
    dataset_name: str
    teacher_importances: List[FeatureImportanceItem]
    student_importances: List[FeatureImportanceItem]
    preservation_score: float
    preservation_status: str
    fairness: Optional[Dict[str, Any]] = None


# Edge Simulator
class EdgeSimulateRequest(BaseModel):
    device_id: str = "rpi4"
    model_size_kb: float = 142.0
    latency_mean_ms: float = 1.15


class EdgeSimulateResponse(BaseModel):
    device_id: str
    device_name: str
    ram_mb: float
    model_ram_mb: float
    ram_utilization_pct: float
    ram_budget_passed: bool
    latency_budget_ms: float
    edge_latency_ms: float
    latency_budget_passed: bool
    estimated_power_watts: float
    overall_status: str               # "PASS" (Lime) or "FAIL" (Magenta)
    limiting_factor: Optional[str] = None


# Drift Monitoring
class MonitorSimulateRequest(BaseModel):
    dataset_name: str = "credit_card_fraud"
    drift_level: float = 0.35


# AI Assistant
class ChatMessage(BaseModel):
    role: str   # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    dataset_name: Optional[str] = "credit_card_fraud"


class ChatResponse(BaseModel):
    reply: str
    citations: List[Dict[str, str]]
    suggested_action: Optional[Dict[str, Any]] = None
