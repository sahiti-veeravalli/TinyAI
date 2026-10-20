import os
import requests
from typing import Dict, Any, List, Optional


class TinyAIClient:
    """Official Python SDK client for TinyAI Edge Intelligence Platform."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        """Check system status, active experiments, and hardware specs."""
        res = requests.get(f"{self.base_url}/api/health")
        res.raise_for_status()
        return res.json()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List ingested datasets."""
        res = requests.get(f"{self.base_url}/api/datasets")
        res.raise_for_status()
        return res.json()

    def load_sample_dataset(self, sample_id: str = "fraud_detection_benchmark") -> Dict[str, Any]:
        """Load a built-in canonical dataset."""
        res = requests.post(
            f"{self.base_url}/api/datasets/load-sample",
            json={"sample_id": sample_id}
        )
        res.raise_for_status()
        return res.json()

    def validate_dataset(self, dataset_id: str, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """Validate schema, missing values, class imbalance, and data leakage."""
        res = requests.post(
            f"{self.base_url}/api/datasets/validate",
            json={"dataset_id": dataset_id, "target_column": target_column, "task_type": task_type}
        )
        res.raise_for_status()
        return res.json()

    def train_baseline(self, dataset_id: str, target_column: str, model_family: str = "lightgbm") -> Dict[str, Any]:
        """Train and benchmark baseline model."""
        res = requests.post(
            f"{self.base_url}/api/baselines/train",
            json={"dataset_id": dataset_id, "target_column": target_column, "model_family": model_family}
        )
        res.raise_for_status()
        return res.json()

    def launch_optimization(
        self,
        dataset_id: str,
        target_column: str,
        baseline_family: str = "lightgbm",
        max_disk_size_kb: float = 250.0,
        max_latency_ms: float = 1.0,
        min_quality_score: float = 0.85,
        priority: str = "balanced"
    ) -> Dict[str, Any]:
        """Launch Pareto frontier optimization experiment across quantization, pruning, and distillation."""
        res = requests.post(
            f"{self.base_url}/api/experiments/launch",
            json={
                "dataset_id": dataset_id,
                "target_column": target_column,
                "baseline_family": baseline_family,
                "max_disk_size_kb": max_disk_size_kb,
                "max_latency_ms": max_latency_ms,
                "min_quality_score": min_quality_score,
                "priority": priority
            }
        )
        res.raise_for_status()
        return res.json()

    def predict(self, features: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, Any]:
        """Run real-time edge prediction."""
        res = requests.post(
            f"{self.base_url}/api/inference/predict",
            json={"features": features, "model_id": model_id}
        )
        res.raise_for_status()
        return res.json()

    def export_bundle(self, model_id: str) -> Dict[str, Any]:
        """Generate standalone edge container bundle with Dockerfile, FastAPI server, and load testing scripts."""
        res = requests.post(f"{self.base_url}/api/models/{model_id}/export")
        res.raise_for_status()
        return res.json()
