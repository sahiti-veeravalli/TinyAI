import pytest
from fastapi.testclient import TestClient
from tinyai.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "platform" in data
    assert "cpu_usage_pct" in data


def test_api_overview(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "projects_count" in data
    assert "hardware_environment" in data


def test_api_datasets_samples(client):
    res = client.get("/api/datasets/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 2


def test_api_load_sample(client):
    res = client.post("/api/datasets/load-sample", json={"sample_id": "edge_iot_predictive_maintenance"})
    assert res.status_code == 200
    data = res.json()
    assert "dataset_id" in data
    assert data["metadata"]["row_count"] > 1000


def test_api_hardware_presets(client):
    res = client.get("/api/hardware-presets")
    assert res.status_code == 200
    presets = res.json()
    assert len(presets) >= 4
    preset_ids = [p["preset_id"] for p in presets]
    assert "rpi4" in preset_ids
    assert "cortex_m" in preset_ids
