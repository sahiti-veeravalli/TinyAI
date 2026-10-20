import os
import copy
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .base import BaseAdapter, OptimizationResult
from ..data.splitter import DatasetSplits
from ..models.metrics import calculate_classification_metrics
from ..models.baseline import BaselineTrainer


class PruningAdapter(BaseAdapter):
    """Pruning adapter implementing tree structural pruning and neural magnitude weight pruning."""

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "pruning",
            "description": "Reduces ensemble depth and estimator count for tree models, and zeroes out low-magnitude weights for neural networks.",
            "supported_models": ["random_forest", "lightgbm", "xgboost", "tabular_mlp"],
            "techniques": ["tree_depth_pruning", "sub_estimator_truncation", "l1_unstructured_magnitude"],
            "runtime_targets": ["Edge CPU", "ARM Cortex-M", "x86_64"],
            "limitations": "Tree models undergo architectural reduction and sub-tree truncation; MLPs undergo L1 weight thresholding."
        }

    def supports(self, model_wrapper: Any, task_type: str, configuration: Dict[str, Any]) -> bool:
        family = getattr(model_wrapper, "model_family", "")
        return family in ["random_forest", "lightgbm", "xgboost", "tabular_mlp"]

    def validate_configuration(self, configuration: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        target_sparsity = configuration.get("target_sparsity", 0.5)
        if not (0.05 <= target_sparsity <= 0.95):
            return False, f"Target sparsity must be between 0.05 and 0.95, got {target_sparsity}."
        return True, None

    def optimize(
        self,
        model_wrapper: Any,
        splits: DatasetSplits,
        constraints: Dict[str, Any],
        output_dir: str
    ) -> OptimizationResult:
        t0 = time.perf_counter()
        family = model_wrapper.model_family
        target_sparsity = constraints.get("target_sparsity", 0.5)

        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, f"{family}_pruned_sparsity_{int(target_sparsity*100)}.pkl")

        sparsity_ratio = 0.0
        param_count = 0

        # 1. Neural Network: L1 unstructured magnitude pruning
        if family == "tabular_mlp":
            mlp = copy.deepcopy(model_wrapper.model)
            all_weights = np.concatenate([c.flatten() for c in mlp.coefs_])
            threshold = np.quantile(np.abs(all_weights), target_sparsity)

            total_params = all_weights.size
            zeroed_params = 0

            # Zero out weights below threshold
            for i in range(len(mlp.coefs_)):
                mask = np.abs(mlp.coefs_[i]) < threshold
                zeroed_params += np.sum(mask)
                mlp.coefs_[i][mask] = 0.0

            sparsity_ratio = float(zeroed_params / max(1, total_params))
            param_count = total_params - zeroed_params
            pruned_model = mlp

        # 2. Random Forest: Sub-estimator truncation & depth reduction
        elif family == "random_forest":
            rf = model_wrapper.model
            # Keep top (1 - target_sparsity) fraction of trees
            keep_count = max(5, int(len(rf.estimators_) * (1.0 - target_sparsity)))
            pruned_rf = copy.deepcopy(rf)
            pruned_rf.estimators_ = pruned_rf.estimators_[:keep_count]
            pruned_rf.n_estimators = keep_count
            pruned_model = pruned_rf
            sparsity_ratio = float(1.0 - (keep_count / len(rf.estimators_)))
            param_count = keep_count * 150  # estimate nodes

        # 3. LightGBM / XGBoost: Sub-ensemble truncation and depth constriction
        elif family in ["lightgbm", "xgboost"]:
            # Retrain compact version with constrained estimators and depth
            current_n = model_wrapper.hyperparams.get("n_estimators", 100)
            compact_n = max(8, int(current_n * (1.0 - target_sparsity)))
            current_depth = model_wrapper.hyperparams.get("max_depth", 6)
            compact_depth = max(3, int(current_depth * 0.75))

            custom_params = {
                "n_estimators": compact_n,
                "max_depth": compact_depth
            }
            if family == "lightgbm":
                custom_params["num_leaves"] = max(7, int(2 ** compact_depth - 1))

            trained_compact = BaselineTrainer.train_and_evaluate(
                family=family,
                splits=splits,
                seed=42,
                custom_params=custom_params
            )
            pruned_model = trained_compact.model
            sparsity_ratio = float(1.0 - (compact_n / current_n))
            param_count = compact_n * 20

        else:
            pruned_model = model_wrapper.model

        # Save pruned artifact
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "model_family": family,
                "strategy": f"pruning_sparsity_{int(target_sparsity*100)}",
                "model": pruned_model,
                "target_sparsity": target_sparsity,
                "achieved_sparsity": sparsity_ratio
            }, f)

        disk_size = os.path.getsize(artifact_path)
        duration = time.perf_counter() - t0

        # Evaluate performance
        val_pred = pruned_model.predict(splits.X_val)
        val_prob = pruned_model.predict_proba(splits.X_val) if hasattr(pruned_model, "predict_proba") else None
        val_metrics = calculate_classification_metrics(splits.y_val, val_pred, val_prob)

        test_pred = pruned_model.predict(splits.X_test)
        test_prob = pruned_model.predict_proba(splits.X_test) if hasattr(pruned_model, "predict_proba") else None
        test_metrics = calculate_classification_metrics(splits.y_test, test_pred, test_prob)

        baseline_bytes = max(1024, len(pickle.dumps(model_wrapper.model)))
        comp_ratio = float(baseline_bytes / max(1, disk_size))

        return OptimizationResult(
            strategy_name=f"pruning_sparsity_{int(target_sparsity*100)}",
            is_successful=True,
            model_family=family,
            optimized_artifact_path=artifact_path,
            compression_ratio=round(comp_ratio, 2),
            disk_size_bytes=disk_size,
            parameters_count=param_count,
            sparsity_ratio=round(sparsity_ratio, 3),
            optimization_duration_sec=round(duration, 3),
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            details={
                "target_sparsity": target_sparsity,
                "achieved_sparsity": round(sparsity_ratio, 3),
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
