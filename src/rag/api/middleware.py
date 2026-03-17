"""Middleware stack — execution order has security implications.

Order (outermost → innermost):
1. Security headers (always set, even on errors)
2. Request-ID injection (all subsequent middleware can log it)
3. Structured request/response logging (never logs body)
4. Redis sliding-window rate limiting per API key
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import redis.asyncio as redis
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# Security headers applied to every response
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request context and response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        # Bind to structlog context so all downstream log calls include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log structured request/response metadata — never log bodies."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            # Never log: body, Authorization header, query string parameters
            # (they may contain API keys or PII)
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiting per API key using Redis.

    Health endpoints are excluded from rate limiting.
    """

    _EXEMPT_PREFIXES = ("/health",)

    def __init__(
        self,
        app,  # type: ignore[no-untyped-def]
        *,
        redis_client: redis.Redis,  # type: ignore[type-arg]
        limit: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._redis = redis_client
        self._limit = limit
        self._window = window_seconds

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip rate limiting for health checks
        if any(request.url.path.startswith(p) for p in self._EXEMPT_PREFIXES):
            return await call_next(request)

        # Use API key from Authorization header as the rate limit key.
        # Fall back to client IP for unauthenticated requests (will 401 anyway).
        auth_header = request.headers.get("Authorization", "")
        client_key = auth_header[-16:] if auth_header else (
            request.client.host if request.client else "unknown"
        )
        redis_key = f"rl:{client_key}"

        now = int(time.time())
        window_start = now - self._window

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {str(uuid.uuid4()): now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, self._window)
        results = await pipe.execute()

        request_count = results[2]
        if request_count > self._limit:
            return Response(
                content='{"error": "rate_limit_exceeded", "detail": "Too many requests"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self._window),
                    **_SECURITY_HEADERS,
                },
            )

        return await call_next(request)
