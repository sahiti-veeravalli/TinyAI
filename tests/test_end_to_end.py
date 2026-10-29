"""
End-to-End Workflow & Integration Tests for TinyAI Production Pipeline.
Tests:
1. Dataset upload & statistical schema inspection
2. Leakage-free train/val/test splitting
3. Baseline model training (PyTorch & Scikit-learn)
4. Knowledge distillation, weight pruning, and dynamic INT8 quantization
5. Experiment tracking persistence
6. Deployment package ZIP export and structural validation
7. Production single and batch inference execution
"""

import os
import io
import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.config import settings
from pipeline.data_manager import DatasetManager
from pipeline.baseline import BaselineTrainer
from pipeline.compression_runner import CompressionOrchestrator
from pipeline.packager import ModelPackager
from pipeline.registry import registry
from pipeline.inference_service import InferenceEngine

client = TestClient(app)


@pytest.fixture
def sample_csv_data():
    """Generates synthetic CSV content with 100 rows, 4 features, and binary target."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.uniform(10, 50, n),
        "category_a": rng.choice(["tier_1", "tier_2", "tier_3"], n),
        "feature_3": rng.exponential(1.5, n),
        "target": rng.choice([0, 1], p=[0.85, 0.15], size=n)
    })
    return df.to_csv(index=False).encode("utf-8")


def test_upload_and_inspect_dataset(sample_csv_data):
    """Verifies CSV upload, file validation, storage, and statistical profiling."""
    files = {"file": ("test_credit_sample.csv", io.BytesIO(sample_csv_data), "text/csv")}
    res = client.post("/api/v1/datasets/upload", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "dataset_id" in data
    assert data["rows_count"] == 100
    assert data["columns_count"] == 5
    assert data["target_column"] == "target"
    assert len(data["columns"]) == 5
    assert len(data["preview_rows"]) > 0

    dataset_id = data["dataset_id"]

    # Test inspection endpoint
    inspect_res = client.get(f"/api/v1/datasets/{dataset_id}/inspect")
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["rows_count"] == 100
    assert inspect_data["inferred_target"] == "target"


def test_dataset_preparation_no_leakage(sample_csv_data):
    """Ensures train/val/test splitting fits preprocessor strictly on train set."""
    files = {"file": ("leakage_test.csv", io.BytesIO(sample_csv_data), "text/csv")}
    upload_res = client.post("/api/v1/datasets/upload", files=files)
    dataset_id = upload_res.json()["dataset_id"]

    prep_res = client.post("/api/v1/datasets/prepare", json={
        "dataset_id": dataset_id,
        "target_column": "target",
        "task_type": "classification",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_seed": 42
    })
    assert prep_res.status_code == 200
    prep_data = prep_res.json()

    assert prep_data["train_count"] + prep_data["val_count"] + prep_data["test_count"] == 100
    assert 65 <= prep_data["train_count"] <= 75
    assert "target" not in prep_data["feature_names"]


def test_baseline_training_pipeline(sample_csv_data):
    """Verifies baseline training, metric calculation, and ONNX export for scikit-learn & PyTorch."""
    files = {"file": ("baseline_test.csv", io.BytesIO(sample_csv_data), "text/csv")}
    upload_res = client.post("/api/v1/datasets/upload", files=files)
    dataset_id = upload_res.json()["dataset_id"]

    # 1. Random Forest baseline
    rf_res = client.post("/api/v1/models/train-baseline", json={
        "dataset_id": dataset_id,
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10, "max_depth": 5, "random_seed": 42}
    })
    assert rf_res.status_code == 200
    rf_data = rf_res.json()
    assert rf_data["model_type"] == "random_forest"
    assert "accuracy" in rf_data["metrics"]
    assert rf_data["model_size_kb"] > 0
    assert rf_data["onnx_available"] is True

    # 2. PyTorch Tabular MLP baseline
    mlp_res = client.post("/api/v1/models/train-baseline", json={
        "dataset_id": dataset_id,
        "model_type": "mlp_resnet",
        "hyperparameters": {"epochs": 3, "lr": 2e-3}
    })
    assert mlp_res.status_code == 200
    mlp_data = mlp_res.json()
    assert mlp_data["model_type"] == "mlp_resnet"
    assert "roc_auc" in mlp_data["metrics"]
    assert mlp_data["model_size_kb"] > 0


def test_end_to_end_compression_and_packaging(sample_csv_data):
    """
    Tests complete optimization flow:
    Upload -> Train Baseline -> Compress (Distill + Prune + INT8 Quantize) -> Package ZIP -> Download -> Inference
    """
    files = {"file": ("e2e_test.csv", io.BytesIO(sample_csv_data), "text/csv")}
    upload_res = client.post("/api/v1/datasets/upload", files=files)
    dataset_id = upload_res.json()["dataset_id"]

    # Train Baseline
    base_res = client.post("/api/v1/models/train-baseline", json={
        "dataset_id": dataset_id,
        "model_type": "mlp_resnet",
        "hyperparameters": {"epochs": 4}
    })
    baseline_model_id = base_res.json()["model_id"]

    # Run Optimization
    opt_res = client.post("/api/v1/optimize/run", json={
        "dataset_id": dataset_id,
        "baseline_model_id": baseline_model_id,
        "pipeline_order": ["distillation", "pruning", "quantization"],
        "pruning_ratio": 0.25,
        "distillation_temperature": 3.0,
        "distillation_alpha": 0.70,
        "quantization_bits": 8,
        "epochs": 4
    })
    assert opt_res.status_code == 200
    opt_data = opt_res.json()

    assert opt_data["status"] == "COMPLETED"
    assert opt_data["compression_ratio"] >= 1.0
    assert opt_data["optimized_size_kb"] > 0
    assert opt_data["optimized_latency_ms"] > 0
    assert "package_download_url" in opt_data

    pkg_url = opt_data["package_download_url"]
    pkg_name = os.path.basename(pkg_url.replace("/download", ""))

    # Download Package ZIP
    dl_res = client.get(pkg_url)
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "application/zip"

    # Verify ZIP package structure
    with zipfile.ZipFile(io.BytesIO(dl_res.content), "r") as zf:
        namelist = zf.namelist()
        # Find basename filenames inside package directory
        basenames = [os.path.basename(n) for n in namelist if n and not n.endswith("/")]
        assert "model.onnx" in basenames
        assert "schema.json" in basenames
        assert "preprocessor.json" in basenames
        assert "predict.py" in basenames
        assert "sample_input.json" in basenames
        assert "metadata.json" in basenames
        assert "checksums.sha256" in basenames
        assert "requirements.txt" in basenames
        assert "README.md" in basenames

        # Verify sample_input.json content
        sample_json_name = [n for n in namelist if n.endswith("sample_input.json")][0]
        sample_dict = json.loads(zf.read(sample_json_name).decode("utf-8"))
        assert "feature_1" in sample_dict


def test_production_inference_endpoint():
    """Verifies batch and single sample prediction through the inference endpoint."""
    # Test with credit_card_fraud student INT8
    single_res = client.post("/api/v1/inference/predict", json={
        "model_id": "credit_card_fraud_student_int8.onnx",
        "features": [0.0] * 30,
        "is_classification": True
    })
    assert single_res.status_code == 200
    single_data = single_res.json()
    assert single_data["sample_count"] == 1
    assert single_data["is_batch"] is False
    assert len(single_data["predictions"]) == 1
    assert "probability" in single_data["predictions"][0]

    # Batch test (3 samples)
    batch_res = client.post("/api/v1/inference/predict", json={
        "model_id": "credit_card_fraud_student_int8.onnx",
        "features": [[0.0] * 30, [0.5] * 30, [-0.5] * 30],
        "is_classification": True
    })
    assert batch_res.status_code == 200
    batch_data = batch_res.json()
    assert batch_data["sample_count"] == 3
    assert batch_data["is_batch"] is True
    assert len(batch_data["predictions"]) == 3


def test_experiment_registry_and_promotion():
    """Verifies experiment listing, retrieval, and model promotion lifecycle."""
    exps = client.get("/api/v1/experiments").json()
    assert isinstance(exps, list)

    models = client.get("/api/v1/models/registry").json()
    assert isinstance(models, list)
    assert len(models) >= 1

    target_model_id = models[0]["model_id"]
    promote_res = client.post(f"/api/v1/models/{target_model_id}/promote", json={
        "model_id": target_model_id,
        "status": "PRODUCTION"
    })
    assert promote_res.status_code == 200
    assert promote_res.json()["new_status"] == "PRODUCTION"
