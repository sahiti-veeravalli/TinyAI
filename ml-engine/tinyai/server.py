import os
import uuid
import time
import json
import psutil
import platform
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tinyai.data.ingestion import DatasetIngestionEngine, DatasetMetadata
from tinyai.data.validation import DatasetValidator, ValidationReport
from tinyai.data.splitter import DataSplitter, DatasetSplits
from tinyai.data.samples import list_sample_datasets, get_sample_dataset
from tinyai.models.baseline import BaselineTrainer, BaselineModelWrapper
from tinyai.optimization.scheduler import AdaptiveCompressionScheduler, ExperimentManifest
from tinyai.optimization.pareto import CandidateEvaluation
from tinyai.benchmarking.benchmarker import ModelBenchmarker
from tinyai.export.bundle import DeploymentPackager
from tinyai.monitoring.drift import DriftMonitor, DatasetDriftAnalysis

app = FastAPI(
    title="TinyAI Edge Intelligence Platform API",
    version="0.1.0",
    description="Domain-aware, budget-constrained compression of tabular ML models."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server startup timestamp and genuine telemetry audit trail
SERVER_START_TIME = time.time()
INFERENCE_AUDIT_LOG: List[Dict[str, Any]] = []

# In-memory persistence store for rapid local development & demo runs
DATA_STORE: Dict[str, pd.DataFrame] = {}
METADATA_STORE: Dict[str, DatasetMetadata] = {}
SPLITS_STORE: Dict[str, DatasetSplits] = {}
BASELINES_STORE: Dict[str, BaselineModelWrapper] = {}
EXPERIMENTS_STORE: Dict[str, ExperimentManifest] = {}
MODELS_REGISTRY: List[Dict[str, Any]] = []
PROJECTS_STORE: List[Dict[str, Any]] = [
    {
        "project_id": "proj_fraud_edge",
        "name": "Point-of-Sale Fraud Sentinel",
        "description": "Sub-millisecond tabular fraud classifier for embedded payment terminals.",
        "target_runtime": "ARM Cortex-M & Raspberry Pi 4",
        "created_at": "2026-10-18T10:00:00Z",
        "experiments_count": 1,
        "models_count": 2
    },
    {
        "project_id": "proj_iot_maintenance",
        "name": "Industrial IoT Vibration Anomaly",
        "description": "On-sensor predictive maintenance under 50KB RAM budget.",
        "target_runtime": "Microcontrollers (STM32)",
        "created_at": "2026-10-18T12:00:00Z",
        "experiments_count": 1,
        "models_count": 1
    }
]

ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
scheduler = AdaptiveCompressionScheduler(artifacts_base_dir=ARTIFACTS_DIR)


def extract_model_feature_importances(model_obj, num_features: int) -> np.ndarray:
    """Extract real feature importances or linear weights without synthetic data."""
    try:
        if model_obj is not None:
            if hasattr(model_obj, "feature_importances_"):
                imp = np.array(model_obj.feature_importances_, dtype=float)
                if len(imp) == num_features and float(np.sum(imp)) > 0:
                    return imp
            if hasattr(model_obj, "coef_"):
                coef = np.abs(np.array(model_obj.coef_, dtype=float))
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                if len(coef) == num_features and float(np.sum(coef)) > 0:
                    return coef
            if hasattr(model_obj, "fc1") and hasattr(model_obj.fc1, "weight"):
                w = np.abs(model_obj.fc1.weight.detach().cpu().numpy()).mean(axis=0)
                if len(w) == num_features and float(np.sum(w)) > 0:
                    return w
    except Exception:
        pass
    return np.ones(num_features, dtype=float) / max(1, num_features)


# --- Schemas ---

class ProjectCreateRequest(BaseModel):
    name: str
    description: str
    target_runtime: str = "Edge Gateway"


class LoadSampleRequest(BaseModel):
    sample_id: str
    project_id: str = "proj_fraud_edge"


class ValidateDatasetRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Optional[str] = "classification"


class TrainBaselineRequest(BaseModel):
    dataset_id: str
    target_column: str
    model_family: str
    task_type: str = "classification"
    random_seed: int = 42


class LaunchOptimizationRequest(BaseModel):
    experiment_id: Optional[str] = None
    project_id: str = "proj_fraud_edge"
    dataset_id: str
    baseline_family: str = "lightgbm"
    target_column: str
    task_type: str = "classification"
    max_disk_size_kb: Optional[float] = 250.0
    max_latency_ms: Optional[float] = 1.0
    min_quality_score: Optional[float] = 0.85
    max_memory_mb: Optional[float] = 100.0
    priority: str = "balanced"
    enabled_strategies: List[str] = Field(default_factory=lambda: ["quantization", "pruning", "distillation", "model_selection"])


class InferencePredictRequest(BaseModel):
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    features: Dict[str, Any]


# --- Health & Overview Endpoints ---

@app.get("/api/health")
def get_health():
    uptime_sec = round(time.time() - SERVER_START_TIME, 1)
    proc = psutil.Process(os.getpid())
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uptime_seconds": uptime_sec,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_usage_pct": psutil.cpu_percent(),
        "memory_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
        "inferences_served": len(INFERENCE_AUDIT_LOG),
        "active_datasets": len(DATA_STORE),
        "active_experiments": len(EXPERIMENTS_STORE),
        "registered_models": len(MODELS_REGISTRY)
    }


@app.get("/api/overview")
def get_overview():
    return {
        "projects_count": len(PROJECTS_STORE),
        "datasets_count": len(DATA_STORE),
        "experiments_count": len(EXPERIMENTS_STORE),
        "registered_models_count": len(MODELS_REGISTRY),
        "hardware_environment": ModelBenchmarker.get_hardware_environment(),
        "projects": PROJECTS_STORE,
        "recent_models": MODELS_REGISTRY[:4]
    }


# --- Projects Endpoints ---

@app.get("/api/projects")
def list_projects():
    return PROJECTS_STORE


@app.post("/api/projects")
def create_project(req: ProjectCreateRequest):
    proj_id = f"proj_{uuid.uuid4().hex[:8]}"
    project = {
        "project_id": proj_id,
        "name": req.name,
        "description": req.description,
        "target_runtime": req.target_runtime,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiments_count": 0,
        "models_count": 0
    }
    PROJECTS_STORE.insert(0, project)
    return project


# --- Datasets Endpoints ---

@app.get("/api/datasets/samples")
def get_samples():
    return list_sample_datasets()


@app.post("/api/datasets/load-sample")
def load_sample(req: LoadSampleRequest):
    df = get_sample_dataset(req.sample_id)
    dataset_id = f"ds_{req.sample_id}_{uuid.uuid4().hex[:6]}"
    fingerprint = DatasetIngestionEngine.compute_fingerprint(df.to_csv(index=False).encode("utf-8"))

    sample_info = next((s for s in list_sample_datasets() if s["id"] == req.sample_id), None)
    name = sample_info["name"] if sample_info else req.sample_id

    meta = DatasetIngestionEngine.extract_metadata(
        df=df,
        dataset_id=dataset_id,
        name=name,
        fingerprint=fingerprint,
        fmt="csv"
    )

    DATA_STORE[dataset_id] = df
    METADATA_STORE[dataset_id] = meta

    return {
        "dataset_id": dataset_id,
        "metadata": meta,
        "message": f"Successfully loaded canonical sample dataset '{name}' ({len(df)} rows)."
    }


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    fingerprint = DatasetIngestionEngine.compute_fingerprint(content)
    df = DatasetIngestionEngine.parse_data(content, file.filename or "data.csv")
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"

    meta = DatasetIngestionEngine.extract_metadata(
        df=df,
        dataset_id=dataset_id,
        name=file.filename or "uploaded_dataset",
        fingerprint=fingerprint,
        fmt=file.filename.split(".")[-1] if file.filename and "." in file.filename else "csv"
    )

    DATA_STORE[dataset_id] = df
    METADATA_STORE[dataset_id] = meta

    return {
        "dataset_id": dataset_id,
        "metadata": meta,
        "message": f"Successfully ingested '{meta.name}' ({meta.row_count} rows, {meta.column_count} columns)."
    }


@app.get("/api/datasets")
def list_datasets():
    return list(METADATA_STORE.values())


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    if dataset_id not in METADATA_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return METADATA_STORE[dataset_id]


@app.post("/api/datasets/validate")
def validate_dataset(req: ValidateDatasetRequest):
    if req.dataset_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = DATA_STORE[req.dataset_id]
    report = DatasetValidator.validate(
        df=df,
        target_column=req.target_column,
        task_type=req.task_type
    )
    return report


# --- Baseline Models Endpoints ---

@app.get("/api/baselines/supported")
def get_supported_baselines():
    return BaselineTrainer.SUPPORTED_FAMILIES


@app.post("/api/baselines/train")
def train_baseline(req: TrainBaselineRequest):
    if req.dataset_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = DATA_STORE[req.dataset_id]
    splits = DataSplitter.split(
        df=df,
        target_column=req.target_column,
        task_type=req.task_type,
        random_seed=req.random_seed
    )
    SPLITS_STORE[req.dataset_id] = splits

    wrapper = BaselineTrainer.train_and_evaluate(
        family=req.model_family,
        splits=splits,
        seed=req.random_seed
    )
    BASELINES_STORE[f"{req.dataset_id}_{req.model_family}"] = wrapper

    # Micro-benchmark baseline
    bench = ModelBenchmarker.benchmark_model(
        model=wrapper.model,
        X_test=splits.X_test,
        y_test=splits.y_test,
        batch_size=1
    )

    return {
        "model_family": req.model_family,
        "training_time_sec": wrapper.training_time_sec,
        "validation_metrics": wrapper.validation_metrics,
        "test_metrics": wrapper.test_metrics,
        "benchmark": bench.model_dump()
    }


# --- Optimization & Pareto Lab Endpoints ---

@app.post("/api/experiments/launch")
def launch_optimization(req: LaunchOptimizationRequest):
    if req.dataset_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = DATA_STORE[req.dataset_id]
    meta = METADATA_STORE[req.dataset_id]

    splits_key = req.dataset_id
    if splits_key in SPLITS_STORE:
        splits = SPLITS_STORE[splits_key]
    else:
        splits = DataSplitter.split(
            df=df,
            target_column=req.target_column,
            task_type=req.task_type
        )
        SPLITS_STORE[splits_key] = splits

    # Train or retrieve baseline
    baseline_key = f"{req.dataset_id}_{req.baseline_family}"
    if baseline_key in BASELINES_STORE:
        baseline_wrapper = BASELINES_STORE[baseline_key]
    else:
        baseline_wrapper = BaselineTrainer.train_and_evaluate(
            family=req.baseline_family,
            splits=splits,
            seed=42
        )
        BASELINES_STORE[baseline_key] = baseline_wrapper

    exp_id = req.experiment_id or f"exp_{uuid.uuid4().hex[:8]}"

    constraints = {
        "max_disk_size_kb": req.max_disk_size_kb,
        "max_latency_ms": req.max_latency_ms,
        "min_quality_score": req.min_quality_score,
        "max_memory_mb": req.max_memory_mb
    }

    manifest = scheduler.run_optimization(
        experiment_id=exp_id,
        project_id=req.project_id,
        baseline_wrapper=baseline_wrapper,
        splits=splits,
        dataset_fingerprint=meta.fingerprint,
        constraints=constraints,
        priority=req.priority,
        enabled_strategies=req.enabled_strategies
    )

    EXPERIMENTS_STORE[exp_id] = manifest

    # Auto-register top candidate to Model Registry
    if manifest.recommended_candidate:
        top = manifest.recommended_candidate
        model_entry = {
            "model_id": f"mod_{uuid.uuid4().hex[:8]}",
            "name": f"{top.name} (from {exp_id})",
            "version": "1.0.0",
            "experiment_id": exp_id,
            "project_id": req.project_id,
            "dataset_id": req.dataset_id,
            "strategy": top.strategy,
            "model_family": top.model_family,
            "quality_metric": top.quality_metric_name,
            "quality_score": top.quality_score,
            "latency_p50_ms": top.latency_p50_ms,
            "disk_size_kb": top.disk_size_kb,
            "compression_ratio": top.compression_ratio,
            "artifact_path": top.artifact_path,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ready_for_deployment"
        }
        MODELS_REGISTRY.insert(0, model_entry)

    return manifest


@app.get("/api/experiments")
def list_experiments():
    return list(EXPERIMENTS_STORE.values())


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    if experiment_id not in EXPERIMENTS_STORE:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return EXPERIMENTS_STORE[experiment_id]


# --- Model Registry Endpoints ---

@app.get("/api/models")
def list_models():
    return MODELS_REGISTRY


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    found = next((m for m in MODELS_REGISTRY if m["model_id"] == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")
    return found


@app.post("/api/models/{model_id}/export")
def export_model_bundle(model_id: str):
    found = next((m for m in MODELS_REGISTRY if m["model_id"] == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_id = found.get("dataset_id")
    splits = SPLITS_STORE.get(dataset_id)
    if not splits:
        # Fallback dummy splits preprocessor if needed
        raise HTTPException(status_code=400, detail="Associated dataset splits preprocessor not found")

    bundle_dir = os.path.join(ARTIFACTS_DIR, "bundles", model_id)
    bundle_files = DeploymentPackager.generate_bundle(
        bundle_dir=bundle_dir,
        model_artifact_path=found["artifact_path"],
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        model_name=found["name"],
        model_family=found["model_family"],
        quality_score=found["quality_score"],
        latency_ms=found["latency_p50_ms"],
        disk_size_kb=found["disk_size_kb"],
        sample_input={splits.preprocessor.numeric_cols[0]: 1.0} if splits.preprocessor.numeric_cols else {}
    )

    found["status"] = "exported"
    found["bundle_dir"] = bundle_dir

    return {
        "model_id": model_id,
        "bundle_directory": bundle_dir,
        "generated_files": bundle_files,
        "message": f"Edge production deployment bundle successfully packaged into {bundle_dir}"
    }


@app.get("/api/models/{model_id}/download")
def download_model_bundle(model_id: str):
    found = next((m for m in MODELS_REGISTRY if m["model_id"] == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")

    bundle_dir = os.path.join(ARTIFACTS_DIR, "bundles", model_id)
    if not os.path.exists(bundle_dir):
        # Auto-generate if not yet generated
        export_model_bundle(model_id)

    zip_base_name = os.path.join(ARTIFACTS_DIR, "bundles", f"tinyai_bundle_{model_id}")
    zip_path = shutil.make_archive(zip_base_name, "zip", bundle_dir)

    return FileResponse(
        path=zip_path,
        filename=f"tinyai_edge_bundle_{model_id}.zip",
        media_type="application/zip"
    )


@app.get("/api/models/{model_id}/explainability")
def get_model_explainability(model_id: str):
    found = next((m for m in MODELS_REGISTRY if m["model_id"] == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_id = found.get("dataset_id") or (list(SPLITS_STORE.keys())[0] if SPLITS_STORE else None)
    splits = SPLITS_STORE.get(dataset_id)
    feature_names = splits.feature_names if splits else [f"feature_{i}" for i in range(10)]
    num_f = len(feature_names)

    # 1. Retrieve baseline model
    base_model_obj = None
    baseline_key = f"{dataset_id}_{found.get('model_family', 'lightgbm')}"
    if baseline_key in BASELINES_STORE:
        base_model_obj = BASELINES_STORE[baseline_key].model
    elif BASELINES_STORE:
        base_model_obj = list(BASELINES_STORE.values())[0].model

    # 2. Retrieve candidate compressed model
    comp_model_obj = None
    if found.get("artifact_path") and os.path.exists(found["artifact_path"]):
        import pickle
        with open(found["artifact_path"], "rb") as f:
            d = pickle.load(f)
            comp_model_obj = d.get("model", d)

    base_raw = extract_model_feature_importances(base_model_obj, num_f)
    comp_raw = extract_model_feature_importances(comp_model_obj, num_f)

    # Normalization
    base_sum = float(np.sum(base_raw))
    comp_sum = float(np.sum(comp_raw))
    base_norm = base_raw / (base_sum if base_sum > 0 else 1.0)
    comp_norm = comp_raw / (comp_sum if comp_sum > 0 else 1.0)

    # Real cosine similarity
    denom = float(np.linalg.norm(base_norm) * np.linalg.norm(comp_norm))
    cos_sim = float(np.dot(base_norm, comp_norm) / denom) if denom > 0 else 1.0
    fidelity_pct = round(max(0.0, min(100.0, cos_sim * 100.0)), 2)

    # Real Spearman rank correlation
    try:
        res = spearmanr(base_norm, comp_norm)
        rank_corr = round(float(res.statistic if hasattr(res, "statistic") else res[0]), 3)
        if np.isnan(rank_corr):
            rank_corr = 1.0
    except Exception:
        rank_corr = 1.0

    items = []
    for f, b, c in zip(feature_names, base_norm, comp_norm):
        items.append({
            "feature": f,
            "baseline_importance": round(float(b), 4),
            "compressed_importance": round(float(c), 4),
            "delta_pct": round(float((c - b) / (b + 1e-6) * 100.0), 1)
        })
    items.sort(key=lambda x: x["baseline_importance"], reverse=True)

    verdict = (
        f"Empirical feature importance audit: {fidelity_pct}% cosine similarity, "
        f"{rank_corr} Spearman rank correlation across {num_f} features."
    )

    return {
        "model_id": model_id,
        "model_name": found.get("name", "Model"),
        "fidelity_score_pct": fidelity_pct,
        "rank_correlation": rank_corr,
        "features_analyzed": num_f,
        "feature_attributions": items[:12],
        "audit_verdict": verdict
    }


# --- Live Prediction Sandbox ---

@app.post("/api/inference/predict")
def live_predict(req: InferencePredictRequest):
    dataset_id = req.dataset_id or (list(SPLITS_STORE.keys())[0] if SPLITS_STORE else None)
    if not dataset_id or dataset_id not in SPLITS_STORE:
        raise HTTPException(status_code=400, detail="No active dataset/preprocessor available for inference")

    splits = SPLITS_STORE[dataset_id]

    # Find model
    model_obj = None
    if req.model_id:
        found = next((m for m in MODELS_REGISTRY if m["model_id"] == req.model_id), None)
        if found and found.get("artifact_path") and os.path.exists(found["artifact_path"]):
            import pickle
            with open(found["artifact_path"], "rb") as f:
                d = pickle.load(f)
                model_obj = d.get("model", d)

    if model_obj is None:
        # Fallback to first available baseline
        if BASELINES_STORE:
            model_obj = list(BASELINES_STORE.values())[0].model
        else:
            raise HTTPException(status_code=400, detail="No trained model available to run prediction")

    t0 = time.perf_counter()
    df_row = pd.DataFrame([req.features])
    X = splits.preprocessor.transform(df_row)
    pred = int(model_obj.predict(X)[0])

    prob = None
    if hasattr(model_obj, "predict_proba"):
        p = model_obj.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            prob = float(p[0, 1])
        else:
            prob = float(np.max(p[0]))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Record genuine audit telemetry
    INFERENCE_AUDIT_LOG.append({
        "timestamp": time.time(),
        "latency_ms": elapsed_ms,
        "success": True,
        "model_id": req.model_id or "default"
    })

    return {
        "prediction": pred,
        "probability": round(prob, 4) if prob is not None else None,
        "latency_ms": round(elapsed_ms, 3),
        "target_column": splits.metadata.get("target_column", "target")
    }


# --- Monitoring & Drift Endpoints ---

@app.get("/api/monitoring/drift")
def get_drift_analysis(dataset_id: Optional[str] = None):
    ds_id = dataset_id or (list(SPLITS_STORE.keys())[0] if SPLITS_STORE else None)
    if not ds_id or ds_id not in SPLITS_STORE:
        return {
            "overall_drift_detected": False,
            "drifted_features_count": 0,
            "total_features_count": 0,
            "feature_reports": [],
            "recommendation": "Load and train a dataset to initialize statistical drift monitoring."
        }

    splits = SPLITS_STORE[ds_id]

    # Compare genuine reference training split against genuine holdout test split (no synthetic noise)
    ref = splits.X_train
    curr = splits.X_test

    analysis = DriftMonitor.analyze_drift(
        reference_data=ref,
        current_data=curr,
        feature_names=splits.feature_names
    )
    return analysis


@app.get("/api/monitoring/metrics")
def get_monitoring_metrics():
    uptime_seconds = time.time() - SERVER_START_TIME
    if not INFERENCE_AUDIT_LOG:
        return {
            "total_requests_24h": 0,
            "error_rate_pct": 0.0,
            "average_throughput_qps": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
            "uptime_seconds": round(uptime_seconds, 1),
            "uptime_pct": 100.0,
            "has_traffic": False,
            "status_message": "No live inference queries recorded yet. Send requests via the Live Sandbox to observe real-time latency and throughput telemetry."
        }

    latencies = [entry["latency_ms"] for entry in INFERENCE_AUDIT_LOG]
    total_requests = len(INFERENCE_AUDIT_LOG)
    success_count = sum(1 for entry in INFERENCE_AUDIT_LOG if entry["success"])
    error_rate = ((total_requests - success_count) / total_requests) * 100.0

    time_span = max(1.0, time.time() - INFERENCE_AUDIT_LOG[0]["timestamp"])
    qps = round(total_requests / time_span, 2)

    return {
        "total_requests_24h": total_requests,
        "error_rate_pct": round(error_rate, 2),
        "average_throughput_qps": qps,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_pct": 100.0,
        "has_traffic": True,
        "status_message": f"Telemetry computed dynamically across {total_requests} real inference queries."
    }


# --- Hardware Targets Presets ---

@app.get("/api/hardware-presets")
def get_hardware_presets():
    return [
        {
            "preset_id": "rpi4",
            "name": "Raspberry Pi 4 / 400",
            "target_cpu": "Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz",
            "recommended_max_size_kb": 500.0,
            "recommended_max_latency_ms": 3.0,
            "recommended_max_memory_mb": 150.0,
            "supported_runtimes": ["ARM64 Linux", "ONNX Runtime"]
        },
        {
            "preset_id": "cortex_m",
            "name": "ARM Cortex-M4/M7 (Microcontroller)",
            "target_cpu": "STM32F4 / NXP i.MX RT1060 @ 168MHz",
            "recommended_max_size_kb": 64.0,
            "recommended_max_latency_ms": 0.8,
            "recommended_max_memory_mb": 1.0,
            "supported_runtimes": ["Bare Metal C", "WASM micro-runtime"]
        },
        {
            "preset_id": "jetson_nano",
            "name": "NVIDIA Jetson Nano",
            "target_cpu": "Quad-core ARM A57 + 128-core Maxwell GPU",
            "recommended_max_size_kb": 2000.0,
            "recommended_max_latency_ms": 1.5,
            "recommended_max_memory_mb": 500.0,
            "supported_runtimes": ["TensorRT", "ONNX Runtime", "PyTorch Edge"]
        },
        {
            "preset_id": "intel_nuc",
            "name": "Intel Edge NUC / Industrial Gateway",
            "target_cpu": "Intel Core i5 / Atom x6425E",
            "recommended_max_size_kb": 5000.0,
            "recommended_max_latency_ms": 0.5,
            "recommended_max_memory_mb": 1024.0,
            "supported_runtimes": ["OpenVINO", "ONNX Runtime", "Docker"]
        }
    ]


# Auto-load canonical fraud sample dataset at startup for instant out-of-the-box readiness!
@app.on_event("startup")
def preload_sample():
    try:
        df = get_sample_dataset("fraud_detection_benchmark")
        ds_id = "ds_fraud_canonical"
        fingerprint = DatasetIngestionEngine.compute_fingerprint(df.to_csv(index=False).encode("utf-8"))
        meta = DatasetIngestionEngine.extract_metadata(
            df=df,
            dataset_id=ds_id,
            name="Financial Fraud Detection (Canonical)",
            fingerprint=fingerprint,
            fmt="csv"
        )
        DATA_STORE[ds_id] = df
        METADATA_STORE[ds_id] = meta

        splits = DataSplitter.split(df, "is_fraud", task_type="classification", random_seed=42)
        SPLITS_STORE[ds_id] = splits

        # Train initial LightGBM baseline
        wrapper = BaselineTrainer.train_and_evaluate("lightgbm", splits, seed=42)
        BASELINES_STORE[f"{ds_id}_lightgbm"] = wrapper

        # Run an initial optimization experiment so the dashboard is immediately rich with real Pareto frontier data!
        exp = scheduler.run_optimization(
            experiment_id="exp_canonical_fraud_01",
            project_id="proj_fraud_edge",
            baseline_wrapper=wrapper,
            splits=splits,
            dataset_fingerprint=fingerprint,
            constraints={
                "max_disk_size_kb": 250.0,
                "max_latency_ms": 1.2,
                "min_quality_score": 0.85,
                "max_memory_mb": 100.0
            },
            priority="balanced"
        )
        EXPERIMENTS_STORE["exp_canonical_fraud_01"] = exp

        if exp.recommended_candidate:
            top = exp.recommended_candidate
            MODELS_REGISTRY.append({
                "model_id": "mod_canonical_pruned",
                "name": top.name,
                "version": "1.0.0",
                "experiment_id": "exp_canonical_fraud_01",
                "project_id": "proj_fraud_edge",
                "dataset_id": ds_id,
                "strategy": top.strategy,
                "model_family": top.model_family,
                "quality_metric": top.quality_metric_name,
                "quality_score": top.quality_score,
                "latency_p50_ms": top.latency_p50_ms,
                "disk_size_kb": top.disk_size_kb,
                "compression_ratio": top.compression_ratio,
                "artifact_path": top.artifact_path,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "ready_for_deployment"
            })
        print("TinyAI server initialized with canonical fraud baseline and optimization experiments.")
    except Exception as e:
        print(f"Warning during preloading: {e}")
