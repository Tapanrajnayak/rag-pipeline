"""Redis-backed query embedding cache.

Key format: sha256(query_text + model_name + input_type)
TTL: 1 hour (configurable)
Serialisation: JSON array of floats (compact, human-readable in redis-cli)
"""

from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis

from rag.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 3600


class EmbeddingCache:
    """Redis cache for query embeddings.

    Only query embeddings (not document embeddings) are cached:
    - Query embeddings are requested far more often (per user request)
    - Document embeddings are computed once at ingest time
    - Cache key includes model name so model upgrades auto-invalidate
    """

    def __init__(
        self,
        redis_client: redis.Redis,  # type: ignore[type-arg]
        *,
        model: str,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        key_prefix: str = "rag:emb:",
    ) -> None:
        self._redis = redis_client
        self._model = model
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _cache_key(self, query: str, input_type: str) -> str:
        """Deterministic cache key: prefix + hex(sha256(query + model + input_type))."""
        raw = f"{query}\x00{self._model}\x00{input_type}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return f"{self._prefix}{digest}"

    async def get(self, query: str, input_type: str = "search_query") -> list[float] | None:
        """Return cached embedding or None on miss.

        Args:
            query: Original query text.
            input_type: Cohere input type (part of cache key).

        Returns:
            Float vector if cached, None otherwise.
        """
        key = self._cache_key(query, input_type)
        raw = await self._redis.get(key)
        if raw is None:
            logger.debug("embedding_cache_miss", key_suffix=key[-8:])
            return None
        logger.debug("embedding_cache_hit", key_suffix=key[-8:])
        return json.loads(raw)  # type: ignore[no-any-return]

    async def set(
        self,
        query: str,
        embedding: list[float],
        input_type: str = "search_query",
    ) -> None:
        """Store an embedding in the cache with TTL.

        Args:
            query: Original query text.
            embedding: Float vector to cache.
            input_type: Cohere input type (part of cache key).
        """
        key = self._cache_key(query, input_type)
        payload = json.dumps(embedding)
        await self._redis.set(key, payload, ex=self._ttl)
        logger.debug("embedding_cache_set", key_suffix=key[-8:], ttl=self._ttl)
