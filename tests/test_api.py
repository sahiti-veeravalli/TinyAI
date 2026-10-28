"""
Integration Tests for TinyAI FastAPI endpoints.
"""

import pytest
from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TinyAI"


def test_benchmarks_endpoint():
    res = client.get("/api/v1/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert "credit_card_fraud" in data
    assert data["credit_card_fraud"]["teacher"]["roc_auc"] > 0


def test_ablation_endpoint():
    res = client.get("/api/v1/ablation")
    assert res.status_code == 200
    data = res.json()
    assert "credit_card_fraud" in data
    assert len(data["credit_card_fraud"]) >= 5


def test_profile_endpoint():
    res = client.post("/api/v1/datasets/profile", json={"dataset_name": "credit_card_fraud", "cost_asymmetry": 50.0})
    assert res.status_code == 200
    data = res.json()
    assert data["n_features"] == 30
    assert data["imbalance_ratio"] > 10.0


def test_schedule_recommend():
    profile_res = client.post("/api/v1/datasets/profile", json={"dataset_name": "credit_card_fraud", "cost_asymmetry": 50.0})
    profile_data = profile_res.json()
    
    rec_res = client.post("/api/v1/schedule/recommend", json={"profile": profile_data})
    assert rec_res.status_code == 200
    plan = rec_res.json()
    assert plan["stage_order"][0] == "distillation"
    assert len(plan["rationale"]) >= 3


def test_dual_predict():
    res = client.post("/api/v1/predict", json={
        "dataset_name": "credit_card_fraud",
        "features": [0.0] * 30
    })
    assert res.status_code == 200
    data = res.json()
    assert "teacher_probability" in data
    assert "student_probability" in data
    assert "agreement" in data


def test_edge_simulate():
    res = client.post("/api/v1/edge/simulate", json={
        "device_id": "rpi4",
        "model_size_kb": 7.2,
        "latency_mean_ms": 0.05
    })
    assert res.status_code == 200
    data = res.json()
    assert data["overall_status"] == "PASS"


def test_chat_assistant():
    res = client.post("/api/v1/assistant/chat", json={
        "messages": [{"role": "user", "content": "Why did INT8 hurt recall on fraud?"}],
        "dataset_name": "credit_card_fraud"
    })
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["citations"]) > 0
