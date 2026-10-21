import os
import time
import uuid
import datetime
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from pydantic import BaseModel, Field

from ..data.splitter import DatasetSplits
from ..models.baseline import BaselineModelWrapper
from ..adapters.quantization import QuantizationAdapter
from ..adapters.pruning import PruningAdapter
from ..adapters.distillation import DistillationAdapter
from ..adapters.model_selection import ModelSelectionAdapter
from ..benchmarking.benchmarker import ModelBenchmarker, BenchmarkReport
from .pareto import CandidateEvaluation, ParetoAnalyzer


def compute_file_sha256(file_path: Optional[str]) -> Optional[str]:
    """Calculates SHA-256 cryptographic hash of an artifact for verifiable provenance."""
    if not file_path or not os.path.exists(file_path):
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExperimentManifest(BaseModel):
    experiment_id: str
    project_id: str
    created_at: str
    dataset_fingerprint: str
    target_column: str
    task_type: str
    baseline_model_family: str
    random_seed: int
    constraints: Dict[str, Any]
    baseline_benchmark: Dict[str, Any]
    candidates: List[CandidateEvaluation]
    pareto_frontier: List[CandidateEvaluation]
    recommended_candidate: Optional[CandidateEvaluation] = None
    priority: str
    hardware_environment: Dict[str, Any]
    objective_prompt: Optional[str] = None


class AdaptiveCompressionScheduler:
    """Budget-aware adaptive compression scheduler for multi-objective Pareto model optimization."""

    def __init__(self, artifacts_base_dir: str = "./artifacts"):
        self.artifacts_base_dir = artifacts_base_dir
        os.makedirs(artifacts_base_dir, exist_ok=True)

        self.adapters = {
            "quantization": QuantizationAdapter(),
            "pruning": PruningAdapter(),
            "distillation": DistillationAdapter(),
            "model_selection": ModelSelectionAdapter(),
        }

    def run_optimization(
        self,
        experiment_id: str,
        project_id: str,
        baseline_wrapper: BaselineModelWrapper,
        splits: DatasetSplits,
        dataset_fingerprint: str,
        constraints: Dict[str, Any],
        priority: str = "balanced",
        enabled_strategies: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        objective_prompt: Optional[str] = None
    ) -> ExperimentManifest:
        """Executes full multi-strategy compression experiments, measures real benchmarks, and extracts the Pareto frontier."""
        exp_dir = os.path.join(self.artifacts_base_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        # 1. Benchmark Baseline
        if progress_callback:
            progress_callback({
                "stage": "benchmarking_baseline",
                "progress_pct": 10,
                "message": f"Benchmarking baseline {baseline_wrapper.model_family} model..."
            })

        baseline_bench = ModelBenchmarker.benchmark_model(
            model=baseline_wrapper.model,
            X_test=splits.X_test,
            y_test=splits.y_test,
            batch_size=1
        )

        quality_metric = "pr_auc" if "pr_auc" in baseline_bench.eval_metrics else "roc_auc"
        if quality_metric not in baseline_bench.eval_metrics:
            quality_metric = "f1_score"

        baseline_qual = baseline_bench.eval_metrics.get(quality_metric, 0.0)

        # Baseline artifact serialization for full cryptographic provenance
        baseline_art_path = os.path.join(exp_dir, f"{experiment_id}_baseline.pkl")
        with open(baseline_art_path, "wb") as f:
            pickle.dump({
                "model": baseline_wrapper.model,
                "model_family": baseline_wrapper.model_family,
                "strategy": "uncompressed_baseline"
            }, f)
        baseline_sha = compute_file_sha256(baseline_art_path)

        # Baseline as candidate #0
        baseline_cand = CandidateEvaluation(
            candidate_id=f"{experiment_id}_baseline",
            name=f"Baseline ({baseline_wrapper.model_family})",
            strategy="uncompressed_baseline",
            model_family=baseline_wrapper.model_family,
            artifact_path=baseline_art_path,
            artifact_checksum_sha256=baseline_sha,
            quality_metric_name=quality_metric,
            quality_score=baseline_qual,
            latency_p50_ms=baseline_bench.warm_latency_ms_p50,
            disk_size_kb=round(baseline_bench.artifact_size_bytes / 1024.0, 2),
            memory_rss_mb=baseline_bench.peak_rss_mb,
            compression_ratio=1.0,
            benchmark_details=baseline_bench.model_dump()
        )
        is_feas, viols = ParetoAnalyzer.check_feasibility(baseline_cand, constraints)
        baseline_cand.is_feasible = is_feas
        baseline_cand.constraint_violations = viols

        candidates: List[CandidateEvaluation] = [baseline_cand]

        # Strategies to run
        strategies = enabled_strategies or ["quantization", "pruning", "distillation", "model_selection"]
        total_steps = len(strategies) + 2
        step_idx = 1

        # 2. Strategy: Quantization (float16)
        if "quantization" in strategies:
            step_idx += 1
            if progress_callback:
                progress_callback({
                    "stage": "quantization",
                    "progress_pct": int((step_idx / total_steps) * 100),
                    "message": "Applying weight & threshold quantization..."
                })

            adapter = self.adapters["quantization"]
            if adapter.supports(baseline_wrapper, "classification", {}):
                res = adapter.optimize(
                    model_wrapper=baseline_wrapper,
                    splits=splits,
                    constraints={"precision": "float16"},
                    output_dir=exp_dir
                )
                if res.is_successful:
                    with open(res.optimized_artifact_path, "rb") as f:
                        q_data = pickle.load(f)
                    bench = ModelBenchmarker.benchmark_model(
                        model=q_data["model"],
                        X_test=splits.X_test,
                        y_test=splits.y_test,
                        artifact_path=res.optimized_artifact_path
                    )
                    cand = CandidateEvaluation(
                        candidate_id=f"{experiment_id}_quant_fp16",
                        name="Quantized (Float16)",
                        strategy=res.strategy_name,
                        model_family=baseline_wrapper.model_family,
                        artifact_path=res.optimized_artifact_path,
                        artifact_checksum_sha256=compute_file_sha256(res.optimized_artifact_path),
                        quality_metric_name=quality_metric,
                        quality_score=bench.eval_metrics.get(quality_metric, 0.0),
                        latency_p50_ms=bench.warm_latency_ms_p50,
                        disk_size_kb=round(bench.artifact_size_bytes / 1024.0, 2),
                        memory_rss_mb=bench.peak_rss_mb,
                        compression_ratio=res.compression_ratio,
                        benchmark_details=bench.model_dump()
                    )
                    is_f, v = ParetoAnalyzer.check_feasibility(cand, constraints)
                    cand.is_feasible = is_f
                    cand.constraint_violations = v
                    candidates.append(cand)

        # 3. Strategy: Pruning (Moderate & Aggressive Sparsity)
        if "pruning" in strategies:
            step_idx += 1
            if progress_callback:
                progress_callback({
                    "stage": "pruning",
                    "progress_pct": int((step_idx / total_steps) * 100),
                    "message": "Pruning ensemble trees and model structures..."
                })

            adapter = self.adapters["pruning"]
            if adapter.supports(baseline_wrapper, "classification", {}):
                for sparsity in [0.4, 0.7]:
                    res = adapter.optimize(
                        model_wrapper=baseline_wrapper,
                        splits=splits,
                        constraints={"target_sparsity": sparsity},
                        output_dir=exp_dir
                    )
                    if res.is_successful:
                        with open(res.optimized_artifact_path, "rb") as f:
                            p_data = pickle.load(f)
                        bench = ModelBenchmarker.benchmark_model(
                            model=p_data["model"],
                            X_test=splits.X_test,
                            y_test=splits.y_test,
                            artifact_path=res.optimized_artifact_path
                        )
                        cand = CandidateEvaluation(
                            candidate_id=f"{experiment_id}_pruned_{int(sparsity*100)}",
                            name=f"Pruned ({int(sparsity*100)}% Sparsity)",
                            strategy=res.strategy_name,
                            model_family=baseline_wrapper.model_family,
                            artifact_path=res.optimized_artifact_path,
                            artifact_checksum_sha256=compute_file_sha256(res.optimized_artifact_path),
                            quality_metric_name=quality_metric,
                            quality_score=bench.eval_metrics.get(quality_metric, 0.0),
                            latency_p50_ms=bench.warm_latency_ms_p50,
                            disk_size_kb=round(bench.artifact_size_bytes / 1024.0, 2),
                            memory_rss_mb=bench.peak_rss_mb,
                            compression_ratio=res.compression_ratio,
                            benchmark_details=bench.model_dump()
                        )
                        is_f, v = ParetoAnalyzer.check_feasibility(cand, constraints)
                        cand.is_feasible = is_f
                        cand.constraint_violations = v
                        candidates.append(cand)

        # 4. Strategy: Knowledge Distillation
        if "distillation" in strategies:
            step_idx += 1
            if progress_callback:
                progress_callback({
                    "stage": "distillation",
                    "progress_pct": int((step_idx / total_steps) * 100),
                    "message": "Distilling knowledge into compact student model..."
                })

            adapter = self.adapters["distillation"]
            if adapter.supports(baseline_wrapper, "classification", {}):
                res = adapter.optimize(
                    model_wrapper=baseline_wrapper,
                    splits=splits,
                    constraints={"temperature": 2.5, "alpha": 0.65},
                    output_dir=exp_dir
                )
                if res.is_successful:
                    with open(res.optimized_artifact_path, "rb") as f:
                        d_data = pickle.load(f)
                    bench = ModelBenchmarker.benchmark_model(
                        model=d_data["model"],
                        X_test=splits.X_test,
                        y_test=splits.y_test,
                        artifact_path=res.optimized_artifact_path
                    )
                    cand = CandidateEvaluation(
                        candidate_id=f"{experiment_id}_distilled_student",
                        name="Distilled Student (Shallow MLP)",
                        strategy=res.strategy_name,
                        model_family="student_mlp",
                        artifact_path=res.optimized_artifact_path,
                        artifact_checksum_sha256=compute_file_sha256(res.optimized_artifact_path),
                        quality_metric_name=quality_metric,
                        quality_score=bench.eval_metrics.get(quality_metric, 0.0),
                        latency_p50_ms=bench.warm_latency_ms_p50,
                        disk_size_kb=round(bench.artifact_size_bytes / 1024.0, 2),
                        memory_rss_mb=bench.peak_rss_mb,
                        compression_ratio=res.compression_ratio,
                        benchmark_details=bench.model_dump()
                    )
                    is_f, v = ParetoAnalyzer.check_feasibility(cand, constraints)
                    cand.is_feasible = is_f
                    cand.constraint_violations = v
                    candidates.append(cand)

        # 5. Strategy: Compact Model Selection
        if "model_selection" in strategies:
            step_idx += 1
            if progress_callback:
                progress_callback({
                    "stage": "model_selection",
                    "progress_pct": int((step_idx / total_steps) * 100),
                    "message": "Evaluating intrinsically compact alternative architectures..."
                })

            adapter = self.adapters["model_selection"]
            for cand_type in ["shallow_tree", "l1_sparse_linear"]:
                res = adapter.optimize(
                    model_wrapper=baseline_wrapper,
                    splits=splits,
                    constraints={"candidate_type": cand_type},
                    output_dir=exp_dir
                )
                if res.is_successful:
                    with open(res.optimized_artifact_path, "rb") as f:
                        m_data = pickle.load(f)
                    bench = ModelBenchmarker.benchmark_model(
                        model=m_data["model"],
                        X_test=splits.X_test,
                        y_test=splits.y_test,
                        artifact_path=res.optimized_artifact_path
                    )
                    clean_name = "Shallow Decision Tree" if cand_type == "shallow_tree" else "L1 Sparse Linear"
                    cand = CandidateEvaluation(
                        candidate_id=f"{experiment_id}_{cand_type}",
                        name=clean_name,
                        strategy=res.strategy_name,
                        model_family=cand_type,
                        artifact_path=res.optimized_artifact_path,
                        artifact_checksum_sha256=compute_file_sha256(res.optimized_artifact_path),
                        quality_metric_name=quality_metric,
                        quality_score=bench.eval_metrics.get(quality_metric, 0.0),
                        latency_p50_ms=bench.warm_latency_ms_p50,
                        disk_size_kb=round(bench.artifact_size_bytes / 1024.0, 2),
                        memory_rss_mb=bench.peak_rss_mb,
                        compression_ratio=res.compression_ratio,
                        benchmark_details=bench.model_dump()
                    )
                    is_f, v = ParetoAnalyzer.check_feasibility(cand, constraints)
                    cand.is_feasible = is_f
                    cand.constraint_violations = v
                    candidates.append(cand)

        # 6. Pareto Frontier Extraction & Ranking
        if progress_callback:
            progress_callback({
                "stage": "pareto_analysis",
                "progress_pct": 95,
                "message": "Extracting Pareto frontier and ranking candidate models..."
            })

        pareto_list = ParetoAnalyzer.extract_pareto_frontier(candidates, feasible_only=False)
        top_recommended, ranked_candidates = ParetoAnalyzer.rank_and_recommend(candidates, priority=priority)

        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            project_id=project_id,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            dataset_fingerprint=dataset_fingerprint,
            target_column=splits.metadata.get("target_column", "target"),
            task_type=splits.metadata.get("task_type", "classification"),
            baseline_model_family=baseline_wrapper.model_family,
            random_seed=splits.metadata.get("random_seed", 42),
            constraints=constraints,
            baseline_benchmark=baseline_bench.model_dump(),
            candidates=ranked_candidates,
            pareto_frontier=pareto_list,
            recommended_candidate=top_recommended,
            priority=priority,
            hardware_environment=baseline_bench.environment,
            objective_prompt=objective_prompt
        )

        manifest_path = os.path.join(exp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        if progress_callback:
            progress_callback({
                "stage": "completed",
                "progress_pct": 100,
                "message": f"Optimization experiment complete! Recommended: {top_recommended.name if top_recommended else 'N/A'}"
            })

        return manifest
