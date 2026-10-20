import os
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from .base import BaseAdapter, OptimizationResult
from ..data.splitter import DatasetSplits
from ..models.metrics import calculate_classification_metrics


class ModelSelectionAdapter(BaseAdapter):
    """Explores compact alternative architectures (e.g. shallow decision trees, regularized linear models) under strict edge resource budgets."""

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "model_selection",
            "description": "Evaluates intrinsically compact baseline architectures that achieve high Pareto efficiency without complex compression pipelines.",
            "supported_models": ["all"],
            "techniques": ["shallow_decision_tree", "l1_sparse_logistic", "compact_linear"],
            "runtime_targets": ["Ultra-low-power MCU", "Bare Metal C", "WASM", "Edge Gateway"],
            "limitations": "Linear and shallow tree models have lower hypothesis capacity on highly non-linear tabular manifolds."
        }

    def supports(self, model_wrapper: Any, task_type: str, configuration: Dict[str, Any]) -> bool:
        return True

    def validate_configuration(self, configuration: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        candidate = configuration.get("candidate_type", "shallow_tree")
        if candidate not in ["shallow_tree", "l1_sparse_linear", "compact_random_forest"]:
            return False, f"Unknown candidate type '{candidate}'."
        return True, None

    def optimize(
        self,
        model_wrapper: Any,
        splits: DatasetSplits,
        constraints: Dict[str, Any],
        output_dir: str
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        candidate_type = constraints.get("candidate_type", "shallow_tree")

        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, f"model_selection_{candidate_type}.pkl")

        if candidate_type == "shallow_tree":
            max_depth = constraints.get("max_depth", 4)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(splits.X_train, splits.y_train)
            param_count = model.tree_.node_count

        elif candidate_type == "l1_sparse_linear":
            # L1 penalty induces true feature sparsity (zeros out uninformative weights)
            C = constraints.get("C", 0.5)
            model = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=42)
            model.fit(splits.X_train, splits.y_train)
            zero_weights = np.sum(model.coef_ == 0)
            param_count = model.coef_.size - zero_weights

        else:
            max_depth = 3
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(splits.X_train, splits.y_train)
            param_count = model.tree_.node_count

        # Save artifact
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "model_family": f"compact_{candidate_type}",
                "strategy": f"model_selection_{candidate_type}",
                "model": model,
                "candidate_type": candidate_type
            }, f)

        disk_size = os.path.getsize(artifact_path)
        duration = time.perf_counter() - t0

        val_pred = model.predict(splits.X_val)
        val_prob = model.predict_proba(splits.X_val) if hasattr(model, "predict_proba") else None
        val_metrics = calculate_classification_metrics(splits.y_val, val_pred, val_prob)

        test_pred = model.predict(splits.X_test)
        test_prob = model.predict_proba(splits.X_test) if hasattr(model, "predict_proba") else None
        test_metrics = calculate_classification_metrics(splits.y_test, test_pred, test_prob)

        baseline_bytes = max(1024, len(pickle.dumps(model_wrapper.model)))
        comp_ratio = float(baseline_bytes / max(1, disk_size))

        return OptimizationResult(
            strategy_name=f"model_selection_{candidate_type}",
            is_successful=True,
            model_family=f"compact_{candidate_type}",
            optimized_artifact_path=artifact_path,
            compression_ratio=round(comp_ratio, 2),
            disk_size_bytes=disk_size,
            parameters_count=int(param_count),
            sparsity_ratio=0.0,
            optimization_duration_sec=round(duration, 3),
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            details={
                "candidate_type": candidate_type,
                "baseline_disk_size_bytes": baseline_bytes,
                "saved_bytes": max(0, baseline_bytes - disk_size)
            }
        )

    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = model.predict(X)
        prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        return calculate_classification_metrics(y, pred, prob)

    def export(self, model: Any, destination_path: str) -> str:
        with open(destination_path, "wb") as f:
            pickle.dump(model, f)
        return destination_path
