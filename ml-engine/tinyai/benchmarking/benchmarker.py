import os
import time
import platform
import psutil
from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from ..models.metrics import calculate_classification_metrics


class BenchmarkReport(BaseModel):
    model_name: str
    artifact_size_bytes: int
    parameter_count: int
    warm_latency_ms_p50: float
    warm_latency_ms_p95: float
    warm_latency_ms_p99: float
    mean_latency_ms: float
    cold_start_load_time_ms: float
    throughput_samples_per_sec: float
    peak_rss_mb: float
    batch_size: int
    eval_metrics: Dict[str, float]
    environment: Dict[str, Any]


class ModelBenchmarker:
    """Micro-benchmark laboratory for measuring real latency, peak RSS memory, throughput, and artifact sizes."""

    @classmethod
    def get_hardware_environment(cls) -> Dict[str, Any]:
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }

    @classmethod
    def benchmark_model(
        cls,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        artifact_path: Optional[str] = None,
        parameter_count: int = 0,
        n_warmup: int = 15,
        n_runs: int = 100,
        batch_size: int = 1
    ) -> BenchmarkReport:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        # 1. Measure Cold Load Time
        t_load_start = time.perf_counter()
        if hasattr(model, "predict"):
            _ = model.predict(X_test[:1])
        cold_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

        # 2. Warm up inference engine
        for _ in range(n_warmup):
            _ = model.predict(X_test[:batch_size])

        # 3. Micro-timed inference runs
        timings: List[float] = []
        n_samples = len(X_test)
        indices = np.random.randint(0, max(1, n_samples - batch_size + 1), size=n_runs)

        for idx in indices:
            batch_slice = X_test[idx : idx + batch_size]
            t0 = time.perf_counter()
            _ = model.predict(batch_slice)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # milliseconds

        timings_arr = np.array(timings)
        p50 = float(np.percentile(timings_arr, 50))
        p95 = float(np.percentile(timings_arr, 95))
        p99 = float(np.percentile(timings_arr, 99))
        mean_lat = float(np.mean(timings_arr))

        # Throughput (samples per second)
        total_bench_time_sec = np.sum(timings) / 1000.0
        total_samples_tested = len(timings) * batch_size
        throughput = float(total_samples_tested / max(1e-6, total_bench_time_sec))

        # Peak RSS memory
        mem_after = process.memory_info().rss
        peak_rss_mb = round(max(mem_before, mem_after) / (1024 * 1024), 2)

        # File size on disk
        artifact_size = 0
        if artifact_path and os.path.exists(artifact_path):
            artifact_size = os.path.getsize(artifact_path)
        else:
            import pickle
            try:
                artifact_size = len(pickle.dumps(model))
            except Exception:
                artifact_size = 1024

        # Full test set quality evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        eval_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

        return BenchmarkReport(
            model_name=getattr(model, "__class__", type("")).__name__,
            artifact_size_bytes=artifact_size,
            parameter_count=parameter_count,
            warm_latency_ms_p50=round(p50, 4),
            warm_latency_ms_p95=round(p95, 4),
            warm_latency_ms_p99=round(p99, 4),
            mean_latency_ms=round(mean_lat, 4),
            cold_start_load_time_ms=round(cold_load_time_ms, 3),
            throughput_samples_per_sec=round(throughput, 1),
            peak_rss_mb=peak_rss_mb,
            batch_size=batch_size,
            eval_metrics=eval_metrics,
            environment=cls.get_hardware_environment()
        )
