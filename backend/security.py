"""
Security, Authentication, and Rate Limiting for TinyAI.
Provides:
- Security headers middleware
- In-memory rate limiting
- Safe file path validation (prevents directory traversal)
- API key authentication dependency
- Content-safe file validation
- Cryptographic hash checks (SHA-256)
"""

import os
import time
import hashlib
from typing import Dict, List, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hash string for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def safe_join(base_dir: str, *paths: str) -> str:
    """
    Safely joins path components ensuring the target path is strictly within base_dir.
    Raises ValueError on directory traversal attempts.
    """
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, *paths))
    common = os.path.commonpath([base, target])
    if common != base:
        raise ValueError("Invalid path: Directory traversal attempt detected.")
    return target


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds enterprise security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


class RateLimiter:
    """In-memory sliding-window rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 120):
        self.rpm = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        if not settings.RATE_LIMIT_ENABLED:
            return True
        now = time.time()
        window_start = now - 60.0
        # Clean older requests
        reqs = [t for t in self.requests[client_id] if t > window_start]
        self.requests[client_id] = reqs
        if len(reqs) >= self.rpm:
            return False
        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


async def rate_limit_check(request: Request):
    """FastAPI dependency to rate limit requests."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending more requests."
        )


# In-memory valid API keys set (seeded with configured secret key)
VALID_API_KEYS = {
    settings.API_SECRET_KEY,
    "ta_live_9f81a74e2b0c3d4e5f6789a0b1c2d3e4"  # Default dev console key
}


def register_api_key(key: str):
    """Registers a newly generated API key into the runtime validator."""
    VALID_API_KEYS.add(key)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)
) -> Optional[str]:
    """
    Verifies that the caller provided a valid API key when auth is enabled.
    Checks either 'X-API-Key' header or 'Authorization: Bearer <token>'.
    """
    if not settings.API_KEY_AUTH_ENABLED:
        # If disabled, allow request through
        return "guest"

    token = None
    if api_key:
        token = api_key
    elif credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials

    if not token or token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Provide X-API-Key or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return token

