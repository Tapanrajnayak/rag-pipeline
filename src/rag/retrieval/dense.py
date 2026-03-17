"""Semantic (dense) search — thin wrapper kept for import symmetry."""

from __future__ import annotations

# Dense search is implemented directly in VectorStore.dense_search().
# This module exists so imports like `from rag.retrieval import dense`
# are available if needed for future standalone use or testing.

__all__: list[str] = []
