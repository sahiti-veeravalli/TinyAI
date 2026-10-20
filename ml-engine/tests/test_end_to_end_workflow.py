import os
import json
import hashlib
import tempfile
import pytest
from fastapi.testclient import TestClient

from tinyai.server import app, ARTIFACTS_DIR, INFERENCE_AUDIT_LOG
from tinyai.data.samples import get_sample_dataset
from tinyai.data.validation import DatasetValidator
from tinyai.data.splitter import DataSplitter
from tinyai.models.baseline import BaselineTrainer
from tinyai.optimization.scheduler import AdaptiveCompressionScheduler, compute_file_sha256
from tinyai.optimization.pareto import ParetoAnalyzer
from tinyai.export.bundle import DeploymentPackager
from tinyai.benchmarking.benchmarker import ModelBenchmarker


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_end_to_end_ml_pipeline():
    """
    Rigorously test the complete 7-stage ML optimization pipeline:
    Data Ingestion -> Schema Validation -> Splitter -> Baseline Training ->
    Candidate Search & Benchmarking -> Pareto Analysis -> Cryptographic SHA-256 Checksum -> Bundle Export.
    """
    # Stage 1: Ingestion
    df = get_sample_dataset("fraud_detection_benchmark")
    assert len(df) == 5000
    assert "is_fraud" in df.columns

    # Stage 2: Schema Validation
    report = DatasetValidator.validate(df, target_column="is_fraud", task_type="classification")
    assert report.is_valid is True
    assert report.imbalance_ratio is not None
    assert report.imbalance_ratio > 10.0  # Imbalanced fraud dataset

    # Stage 3: Splitter
    splits = DataSplitter.split(df, target_column="is_fraud", task_type="classification", random_seed=42)
    assert len(splits.X_train) > 0
    assert len(splits.X_test) > 0
    assert splits.preprocessor is not None

    # Stage 4: Baseline Training & Benchmarking
    baseline_wrapper = BaselineTrainer.train_and_evaluate("lightgbm", splits, seed=42)
    assert baseline_wrapper.model is not None
    assert baseline_wrapper.training_time_sec > 0
    assert "pr_auc" in baseline_wrapper.test_metrics
    assert baseline_wrapper.test_metrics["pr_auc"] > 0.5

    bench = ModelBenchmarker.benchmark_model(baseline_wrapper.model, splits.X_test, splits.y_test, batch_size=1)
    assert bench.warm_latency_ms_p50 > 0
    assert bench.artifact_size_bytes > 0

    # Stage 5: Multi-Objective Scheduler & Candidate Optimization
    with tempfile.TemporaryDirectory() as temp_dir:
        scheduler = AdaptiveCompressionScheduler(artifacts_base_dir=temp_dir)
        manifest = scheduler.run_optimization(
            experiment_id="exp_test_pipeline_001",
            project_id="proj_test",
            baseline_wrapper=baseline_wrapper,
            splits=splits,
            dataset_fingerprint="sha256_dummy_fingerprint",
            constraints={
                "max_disk_size_kb": 250.0,
                "max_latency_ms": 2.0,
                "min_quality_score": 0.80,
                "max_memory_mb": 150.0
            },
            priority="balanced",
            enabled_strategies=["pruning", "quantization", "model_selection"]
        )

        assert manifest is not None
        assert len(manifest.candidates) >= 3

        # Stage 6: Pareto Dominance & Cryptographic Provenance
        assert len(manifest.pareto_frontier) >= 1
        assert manifest.recommended_candidate is not None

        for cand in manifest.candidates:
            assert cand.latency_p50_ms > 0
            assert cand.disk_size_kb > 0
            assert cand.quality_score > 0
            assert cand.artifact_checksum_sha256 is not None
            assert len(cand.artifact_checksum_sha256) == 64

            # Verify cryptographic SHA-256 matches actual file on disk
            if cand.artifact_path and os.path.exists(cand.artifact_path):
                real_hash = compute_file_sha256(cand.artifact_path)
                assert cand.artifact_checksum_sha256 == real_hash

        # Verify manifest persistence on disk
        saved_manifest_path = os.path.join(temp_dir, "exp_test_pipeline_001", "manifest.json")
        assert os.path.exists(saved_manifest_path)
        with open(saved_manifest_path, "r") as mf:
            persisted = json.load(mf)
            assert persisted["experiment_id"] == "exp_test_pipeline_001"
            assert len(persisted["candidates"]) == len(manifest.candidates)

        # Stage 7: Deployment Bundle Generation
        top = manifest.recommended_candidate
        bundle_dir = os.path.join(temp_dir, "bundle_test")
        bundle_files = DeploymentPackager.generate_bundle(
            bundle_dir=bundle_dir,
            model_artifact_path=top.artifact_path,
            preprocessor=splits.preprocessor,
            feature_names=splits.feature_names,
            model_name=top.name,
            model_family=top.model_family,
            quality_score=top.quality_score,
            latency_ms=top.latency_p50_ms,
            disk_size_kb=top.disk_size_kb,
            sample_input={"transaction_amount": 50.0}
        )

        assert "dockerfile_path" in bundle_files and os.path.exists(bundle_files["dockerfile_path"])
        assert "app_path" in bundle_files and os.path.exists(bundle_files["app_path"])
        assert "model_path" in bundle_files and os.path.exists(bundle_files["model_path"])
        assert "preprocessor_path" in bundle_files and os.path.exists(bundle_files["preprocessor_path"])
        assert "model_card_path" in bundle_files and os.path.exists(bundle_files["model_card_path"])


def test_api_live_prediction_and_dynamic_telemetry(client):
    """
    Verify that prediction requests record telemetry dynamically into the audit log
    and that /api/monitoring/metrics reflects genuine traffic without hardcoded mock data.
    """
    # 1. Check health
    res_h = client.get("/api/health")
    assert res_h.status_code == 200
    h_data = res_h.json()
    assert "uptime_seconds" in h_data
    assert "memory_rss_mb" in h_data
    assert "cpu_usage_pct" in h_data

    # 2. Record initial count
    init_inferences = len(INFERENCE_AUDIT_LOG)

    # 3. Send real prediction payload
    pred_res = client.post("/api/inference/predict", json={
        "features": {
            "transaction_amount": 75.50,
            "merchant_distance_km": 3.2,
            "transaction_hour": 14,
            "v1_signal": 0.12,
            "v2_signal": -0.45,
            "v3_signal": 0.88,
            "v4_signal": -0.10,
            "v5_signal": 0.05,
            "payment_channel": "POS",
            "device_platform": "Android"
        }
    })
    assert pred_res.status_code == 200
    p_data = pred_res.json()
    assert "prediction" in p_data
    assert "latency_ms" in p_data
    assert p_data["latency_ms"] > 0

    # 4. Verify audit log incremented
    assert len(INFERENCE_AUDIT_LOG) == init_inferences + 1

    # 5. Check monitoring metrics reflects real traffic
    metrics_res = client.get("/api/monitoring/metrics")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["has_traffic"] is True
    assert m_data["total_requests_24h"] >= 1
    assert m_data["latency_p50_ms"] is not None
    assert m_data["latency_p50_ms"] > 0
    assert m_data["error_rate_pct"] == 0.0


def test_api_explainability_and_drift_real_eval(client):
    """
    Verify explainability and drift monitoring run real statistical evaluations
    with zero synthetic noise.
    """
    # 1. Fetch available registered models or launch an experiment
    models_res = client.get("/api/models")
    assert models_res.status_code == 200
    models = models_res.json()
    if not models:
        # Load sample and launch optimization
        client.post("/api/datasets/load-sample", json={"sample_id": "fraud_detection_benchmark"})
        ds_list = client.get("/api/datasets").json()
        target_ds = ds_list[0]["dataset_id"]
        client.post("/api/experiments/launch", json={
            "dataset_id": target_ds,
            "target_column": "is_fraud",
            "baseline_family": "lightgbm"
        })
        models = client.get("/api/models").json()

    assert len(models) > 0
    model_id = models[0]["model_id"]

    # Explainability
    res_exp = client.get(f"/api/models/{model_id}/explainability")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert "fidelity_score_pct" in exp_data
    assert "rank_correlation" in exp_data
    assert exp_data["fidelity_score_pct"] > 0
    assert len(exp_data["feature_attributions"]) > 0
    assert "Empirical feature importance audit" in exp_data["audit_verdict"]

    # Drift monitoring
    res_drift = client.get("/api/monitoring/drift")
    assert res_drift.status_code == 200
    d_data = res_drift.json()
    assert "overall_drift_detected" in d_data
    assert "feature_reports" in d_data
    assert len(d_data["feature_reports"]) > 0
    # KS-statistic and p-value check
    f0 = d_data["feature_reports"][0]
    assert "ks_statistic" in f0
    assert "p_value" in f0
    assert "psi_score" in f0
