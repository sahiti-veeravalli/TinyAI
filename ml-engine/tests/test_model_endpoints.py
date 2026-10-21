import io
import pickle
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from tinyai.server import app

client = TestClient(app)


def test_parse_objective_api():
    payload = {
        "prompt": "Optimize this model for CPU inference. Keep macro F1 above 0.90, keep the model under 20 MB, and minimize median inference latency."
    }
    response = client.post("/api/experiments/parse-objective", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["parser_type"] == "deterministic_rule_based"
    assert data["max_disk_size_kb"] == 20480.0
    assert data["min_quality_score"] == 0.90
    assert data["priority"] == "lowest_latency"
    assert data["is_contradictory"] is False


def test_sample_catalog_and_load_api():
    # 1. Fetch catalog
    res = client.get("/api/models/sample-catalog")
    assert res.status_code == 200
    catalog = res.json()
    assert len(catalog) >= 3
    sample_ids = [c["sample_id"] for c in catalog]
    assert "sample_fraud_lightgbm" in sample_ids

    # 2. Load a sample model
    res_load = client.post("/api/models/load-sample", json={"sample_id": "sample_fraud_lightgbm"})
    assert res_load.status_code == 200
    load_data = res_load.json()
    assert load_data["model"]["model_family"] == "lightgbm"
    assert "Point-of-Sale Fraud" in load_data["model"]["name"]


def test_upload_valid_model_api():
    # Create and serialize a simple fitted scikit-learn model
    clf = LogisticRegression()
    clf.fit([[0.1, 0.2], [0.8, 0.9]], [0, 1])
    bio = io.BytesIO()
    pickle.dump(clf, bio)
    bio.seek(0)

    files = {"file": ("test_model.pkl", bio.getvalue(), "application/octet-stream")}
    res = client.post("/api/models/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["model"]["is_valid"] is True
    assert data["model"]["model_family"] == "logistic_regression"
    assert data["model"]["checksum_sha256"] is not None
    assert len(data["model"]["supported_operations"]) > 0


def test_upload_invalid_format_and_unfitted_model():
    # 1. Bad file extension
    files = {"file": ("invalid.txt", b"plain text", "text/plain")}
    res = client.post("/api/models/upload", files=files)
    assert res.status_code == 400
    assert "Unsupported format" in res.json()["detail"]

    # 2. Upload dict without predict
    bio = io.BytesIO()
    pickle.dump({"not_a_model": 123}, bio)
    files2 = {"file": ("bad_object.pkl", bio.getvalue(), "application/octet-stream")}
    res2 = client.post("/api/models/upload", files=files2)
    assert res2.status_code == 422
    assert "does not provide a callable 'predict()'" in res2.json()["detail"]


def test_export_bundle_with_checksums():
    # Ensure there is a model in registry
    models_res = client.get("/api/models")
    assert models_res.status_code == 200
    models = models_res.json()
    assert len(models) > 0
    mod_id = models[0]["model_id"]

    export_res = client.post(f"/api/models/{mod_id}/export")
    assert export_res.status_code == 200
    exp_data = export_res.json()
    assert "file_checksums" in exp_data
    assert "model.pkl" in exp_data["file_checksums"]
    assert "reproduction_instructions" in exp_data
    assert len(exp_data["reproduction_instructions"]) >= 3
