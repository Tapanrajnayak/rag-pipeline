"""Health check endpoints — liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    qdrant: str
    redis: str


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe — is the process up?",
)
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe. Returns 200 if the process is running."""
    return LivenessResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe — are all dependencies reachable?",
)
async def readiness(request: Request) -> ReadinessResponse:
    """Kubernetes readiness probe. Checks Qdrant and Redis connectivity.

    Returns 200 if all dependencies are healthy.
    Returns 503 if any dependency is unavailable.
    """
    from fastapi.responses import JSONResponse

    vector_store = request.app.state.vector_store
    redis_client = request.app.state.redis_client

    qdrant_ok = await vector_store.health_check()

    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    all_ok = qdrant_ok and redis_ok
    body = ReadinessResponse(
        status="ready" if all_ok else "degraded",
        qdrant="ok" if qdrant_ok else "unavailable",
        redis="ok" if redis_ok else "unavailable",
    )

    if not all_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )
    return body
