import json
import pytest
from fastapi.testclient import TestClient
from tinyai.server import app

client = TestClient(app)


def test_complete_replit_inspired_journey():
    # 1. Health check & host CPU telemetry
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    health = health_res.json()
    assert health["status"] == "healthy"
    assert "platform" in health

    # 2. Sample catalog lookup
    cat_res = client.get("/api/models/sample-catalog")
    assert cat_res.status_code == 200
    catalog = cat_res.json()
    assert len(catalog) >= 3
    sample_fraud = next(c for c in catalog if c["sample_id"] == "sample_fraud_lightgbm")
    assert sample_fraud["framework"] == "LightGBM 4.3"

    # 3. Load sample model
    load_res = client.post("/api/models/load-sample", json={"sample_id": "sample_fraud_lightgbm"})
    assert load_res.status_code == 200
    loaded = load_res.json()
    assert loaded["model"]["is_valid"] is True
    assert loaded["model"]["model_family"] == "lightgbm"

    # 4. Parse conversational objective
    prompt = (
        "Optimize this model for CPU inference. Keep macro F1 above 0.90, "
        "keep the model under 20 MB, and minimize median inference latency."
    )
    parse_res = client.post("/api/experiments/parse-objective", json={"prompt": prompt})
    assert parse_res.status_code == 200
    parsed = parse_res.json()
    assert parsed["parser_type"] == "deterministic_rule_based"
    assert parsed["min_quality_score"] == 0.90
    assert parsed["max_disk_size_kb"] == 20480.0
    assert parsed["priority"] == "lowest_latency"

    # 5. Launch real optimization run
    launch_res = client.post("/api/experiments/launch", json={
        "dataset_id": loaded["dataset_id"],
        "target_column": "is_fraud",
        "baseline_family": loaded["model"]["model_family"],
        "max_disk_size_kb": 250.0,
        "max_latency_ms": 1.2,
        "min_quality_score": 0.85,
        "max_memory_mb": 300.0,
        "priority": "balanced",
        "enabled_strategies": ["quantization", "pruning", "distillation", "model_selection"],
        "objective_prompt": prompt
    })
    assert launch_res.status_code == 200
    manifest = launch_res.json()
    assert len(manifest["candidates"]) >= 4
    assert len(manifest["pareto_frontier"]) >= 1

    # 6. Verify Pareto frontier and recommendation rationale
    top = manifest["recommended_candidate"]
    assert top is not None
    assert top["is_feasible"] is True
    assert "Selected under 'balanced' optimization strategy" in top["recommendation_rationale"]
    assert top["latency_p50_ms"] > 0
    assert top["disk_size_kb"] > 0

    # 7. Export production bundle
    export_res = client.post(f"/api/models/{top['candidate_id']}/export")
    assert export_res.status_code == 200
    bundle = export_res.json()
    assert "file_checksums" in bundle
    assert "model.pkl" in bundle["file_checksums"]
    assert "reproduction_instructions" in bundle
    assert len(bundle["reproduction_instructions"]) >= 3

    # 8. Download bundle zip
    download_res = client.get(f"/api/models/{top['candidate_id']}/download")
    assert download_res.status_code == 200
    assert download_res.headers["content-type"] == "application/zip"
    assert len(download_res.content) > 1000

    # 9. Execute live prediction in sandbox
    pred_res = client.post("/api/inference/predict", json={
        "model_id": top["candidate_id"],
        "features": {
            "amt": 52.3,
            "lat": 34.05,
            "long": -118.25,
            "city_pop": 50000,
            "unix_time": 1325376000,
            "merch_lat": 34.06,
            "merch_long": -118.26
        }
    })
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["prediction"] in [0, 1]
    assert pred_data["latency_ms"] > 0
    assert pred_data["latency_ms"] < 100.0  # sub-millisecond or fast host CPU
