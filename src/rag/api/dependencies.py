"""FastAPI dependency injection — auth, services, and shared clients."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag.core.config import Settings, get_settings
from rag.core.errors import AuthenticationError
from rag.core.security import UserContext, validate_jwt

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> UserContext:
    """Extract and validate JWT from the Authorization: Bearer header.

    Args:
        credentials: Parsed HTTP Bearer credentials.
        settings: Application settings (for JWT config).

    Returns:
        Authenticated UserContext.

    Raises:
        HTTPException 401: if credentials are missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = validate_jwt(
            credentials.credentials,
            secret_key=settings.jwt_secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return user


def get_vector_store(request: Request):  # type: ignore[no-untyped-def]
    """Return VectorStore from app state."""
    return request.app.state.vector_store


def get_doc_repo(request: Request):  # type: ignore[no-untyped-def]
    """Return document repository from app state."""
    return request.app.state.doc_repo


def get_embedder(request: Request):  # type: ignore[no-untyped-def]
    """Return embedding provider from app state."""
    return request.app.state.embedder


def get_sparse_encoder(request: Request):  # type: ignore[no-untyped-def]
    """Return sparse encoder from app state."""
    return request.app.state.sparse_encoder
