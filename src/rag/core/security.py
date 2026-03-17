"""JWT validation, RBAC, and Qdrant ACL filter builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jose import ExpiredSignatureError, JWTError, jwt

from rag.core.errors import AuthenticationError


@dataclass(frozen=True)
class UserContext:
    """Authenticated user context extracted from JWT claims."""

    user_id: str
    groups: frozenset[str]
    email: str | None = None

    @property
    def acl_groups(self) -> list[str]:
        """Groups this user belongs to, always includes 'all'."""
        return sorted(self.groups | {"all"})


def validate_jwt(
    token: str,
    *,
    secret_key: str,
    algorithm: str,
    audience: str,
    issuer: str,
) -> UserContext:
    """Validate a JWT and return the extracted UserContext.

    Raises:
        AuthenticationError: if the token is invalid, expired, or has wrong claims.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            audience=audience,
            issuer=issuer,
        )
    except ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token missing 'sub' claim")

    groups: list[str] = payload.get("groups", [])
    email: str | None = payload.get("email")

    return UserContext(
        user_id=user_id,
        groups=frozenset(groups),
        email=email,
    )


def build_acl_filter(user: UserContext) -> dict[str, Any]:
    """Build a Qdrant filter that restricts results to documents the user can see.

    The filter is applied atomically at the vector store layer — there is no
    application-side window where documents could be retrieved before the ACL
    check runs.

    Args:
        user: Authenticated user context with group memberships.

    Returns:
        Qdrant filter dict. Documents must have at least one ACL group that
        matches the user's groups (including the implicit 'all' group).
    """
    return {
        "must": [
            {
                "key": "acl_groups",
                "match": {"any": user.acl_groups},
            }
        ]
    }


def create_access_token(
    user_id: str,
    groups: list[str],
    *,
    secret_key: str,
    algorithm: str,
    audience: str,
    issuer: str,
    expires_delta_seconds: int = 3600,
    email: str | None = None,
) -> str:
    """Create a signed JWT for testing and demo purposes.

    Args:
        user_id: Subject identifier.
        groups: ACL group memberships.
        secret_key: HMAC signing key.
        algorithm: JWT signing algorithm.
        audience: Expected audience claim.
        issuer: Expected issuer claim.
        expires_delta_seconds: Token lifetime in seconds.
        email: Optional email claim.

    Returns:
        Signed JWT string.
    """
    import time

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "groups": groups,
        "iat": now,
        "exp": now + expires_delta_seconds,
        "aud": audience,
        "iss": issuer,
    }
    if email:
        payload["email"] = email

    return jwt.encode(payload, secret_key, algorithm=algorithm)
