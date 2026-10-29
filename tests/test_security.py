"""
Security, Threat Defense, and Policy Verification Tests for TinyAI.
Tests:
1. Path traversal attacks prevention (models, packages, datasets)
2. Enterprise security response headers
3. Upload size limit enforcement and file type validation
4. Cryptographic integrity checking (SHA-256)
5. Safe join path boundaries
6. Authentication & Rate limiting validation
"""

import os
import io
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.security import safe_join, compute_sha256, compute_file_sha256, RateLimiter
from backend.config import settings

client = TestClient(app)


def test_security_headers_present():
    """Verifies that all responses include strict security headers."""
    res = client.get("/health")
    assert res.status_code == 200
    headers = res.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("x-xss-protection") == "1; mode=block"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_path_traversal_prevention_on_downloads():
    """Ensures malicious path traversal payloads (e.g. ../../etc/passwd) are rejected."""
    # Package download
    res1 = client.get("/api/v1/packages/../../etc/passwd/download")
    assert res1.status_code in (400, 404)

    res2 = client.get("/api/v1/packages/..%2f..%2fconfig.py/download")
    assert res2.status_code in (400, 404)

    # Model download
    res3 = client.get("/api/v1/models/../../config.py/download")
    assert res3.status_code in (400, 404)


def test_safe_join_utility():
    """Verifies safe_join throws ValueError when escaping base directory."""
    base = settings.RESULTS_DIR
    # Valid child
    child = safe_join(base, "models", "test.onnx")
    assert os.path.abspath(child).startswith(os.path.abspath(base))

    # Path traversal attack
    with pytest.raises(ValueError):
        safe_join(base, "..", "..", "windows", "system32")

    with pytest.raises(ValueError):
        safe_join(base, "/etc/passwd")


def test_upload_empty_file_rejected():
    """Rejects empty file upload."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res = client.post("/api/v1/datasets/upload", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_upload_invalid_extension_rejected():
    """Rejects executable or non-tabular extensions."""
    files = {"file": ("malware.exe", io.BytesIO(b"MZfakebinary"), "application/x-msdownload")}
    res = client.post("/api/v1/datasets/upload", files=files)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_upload_oversized_file_rejected():
    """Rejects file exceeding max upload limit."""
    # Test validator with small dummy limit
    from pipeline.data_manager import DatasetManager
    oversized_bytes = b"a" * 1024
    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        DatasetManager.validate_and_save_upload(
            oversized_bytes,
            "big.csv",
            max_size_bytes=512
        )


def test_sha256_hash_computation():
    """Ensures consistent cryptographic hashing."""
    data = b"TinyAI Enterprise Verification"
    h1 = compute_sha256(data)
    h2 = compute_sha256(data)
    assert h1 == h2
    assert len(h1) == 64


def test_rate_limiter_logic():
    """Verifies that in-memory RateLimiter throttles excess requests."""
    limiter = RateLimiter(requests_per_minute=5)
    client_id = "test_ip_123"
    for _ in range(5):
        assert limiter.is_allowed(client_id) is True
    # 6th request should be blocked
    assert limiter.is_allowed(client_id) is False

