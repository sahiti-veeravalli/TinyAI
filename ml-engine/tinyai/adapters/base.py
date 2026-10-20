from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from ..data.splitter import DatasetSplits


class OptimizationResult(BaseModel):
    strategy_name: str
    is_successful: bool
    model_family: str
    optimized_artifact_path: Optional[str] = None
    compression_ratio: float = 1.0
    disk_size_bytes: int = 0
    parameters_count: int = 0
    sparsity_ratio: float = 0.0
    optimization_duration_sec: float = 0.0
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class BaseAdapter(ABC):
    """Common abstraction for all model compression, quantization, pruning, and distillation adapters."""

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Returns the capabilities, supported model families, formats, and edge runtime targets."""
        pass

    @abstractmethod
    def supports(self, model_wrapper: Any, task_type: str, configuration: Dict[str, Any]) -> bool:
        """Returns True if this adapter supports the given model family and task configuration."""
        pass

    @abstractmethod
    def validate_configuration(self, configuration: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates optimization hyperparameters and constraint compatibility."""
        pass

    @abstractmethod
    def optimize(
        self,
        model_wrapper: Any,
        splits: DatasetSplits,
        constraints: Dict[str, Any],
        output_dir: str
    ) -> OptimizationResult:
        """Executes the concrete compression transformation and measures real outcomes."""
        pass

    @abstractmethod
    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates model performance metrics on given features and labels."""
        pass

    @abstractmethod
    def export(self, model: Any, destination_path: str) -> str:
        """Exports the compressed model to disk in its supported runtime format."""
        pass
