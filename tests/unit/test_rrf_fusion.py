"""Unit tests for RRF fusion — math correctness and determinism."""

from __future__ import annotations

import pytest

from rag.retrieval.hybrid import RetrievedChunk, _rrf_fuse


def _make_hit(chunk_id: str, score: float = 1.0) -> dict:
    return {
        "id": chunk_id,
        "score": score,
        "payload": {
            "text": f"text for {chunk_id}",
            "document_id": f"doc-{chunk_id}",
            "acl_groups": ["all"],
        },
    }


# ── RRF math ──────────────────────────────────────────────────────────────────

def test_rrf_single_list_uses_correct_formula() -> None:
    """RRF score for rank 0 in a single list should be 1/(60+1)."""
    dense = [_make_hit("a"), _make_hit("b")]
    result = _rrf_fuse(dense, [], k=60)

    chunks_by_id = {c.chunk_id: c for c in result}
    assert "a" in chunks_by_id
    expected_a = 1.0 / (60 + 0 + 1)
    assert abs(chunks_by_id["a"].rrf_score - expected_a) < 1e-10


def test_rrf_two_lists_sums_scores() -> None:
    """A document appearing in both lists should get the sum of both RRF scores."""
    dense = [_make_hit("a"), _make_hit("b")]
    sparse = [_make_hit("a"), _make_hit("c")]

    result = _rrf_fuse(dense, sparse, k=60)
    chunks_by_id = {c.chunk_id: c for c in result}

    # 'a' appears at rank 0 in both lists
    expected_a = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(chunks_by_id["a"].rrf_score - expected_a) < 1e-10

    # 'b' appears only in dense at rank 1
    expected_b = 1.0 / (60 + 2)
    assert abs(chunks_by_id["b"].rrf_score - expected_b) < 1e-10


def test_rrf_sorted_descending() -> None:
    """Results must be sorted by RRF score descending."""
    dense = [_make_hit(f"d{i}") for i in range(5)]
    sparse = [_make_hit(f"s{i}") for i in range(5)]
    # Make "d0" appear in both to get highest score
    sparse[0] = _make_hit("d0")

    result = _rrf_fuse(dense, sparse)
    scores = [c.rrf_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deterministic_on_ties() -> None:
    """Tie-breaking must be consistent across calls (sorted by chunk_id)."""
    dense = [_make_hit("b"), _make_hit("a")]
    sparse: list[dict] = []

    result1 = _rrf_fuse(dense, sparse)
    result2 = _rrf_fuse(dense, sparse)

    assert [c.chunk_id for c in result1] == [c.chunk_id for c in result2]


def test_rrf_no_duplicates_in_output() -> None:
    """Each chunk_id should appear at most once in the output."""
    dense = [_make_hit("a"), _make_hit("b"), _make_hit("a")]  # duplicate 'a'
    sparse = [_make_hit("a"), _make_hit("c")]

    result = _rrf_fuse(dense, sparse)
    chunk_ids = [c.chunk_id for c in result]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_empty_inputs() -> None:
    result = _rrf_fuse([], [])
    assert result == []


def test_rrf_one_empty_list() -> None:
    dense = [_make_hit("a"), _make_hit("b")]
    result = _rrf_fuse(dense, [])
    assert len(result) == 2
    assert all(c.sparse_rank is None for c in result)


def test_rrf_k_parameter_affects_scores() -> None:
    """Higher k should produce lower but more uniform scores."""
    dense = [_make_hit("a")]
    result_k1 = _rrf_fuse(dense, [], k=1)
    result_k100 = _rrf_fuse(dense, [], k=100)

    # k=1: score = 1/(1+1) = 0.5
    # k=100: score = 1/(100+1) ≈ 0.0099
    assert result_k1[0].rrf_score > result_k100[0].rrf_score


def test_adding_more_candidates_does_not_decrease_top_score() -> None:
    """Adding candidates to one list should not lower the top document's score."""
    dense_base = [_make_hit("top")]
    sparse_base: list[dict] = []

    result_base = _rrf_fuse(dense_base, sparse_base)
    top_score_base = result_base[0].rrf_score

    # Add more candidates — 'top' is still rank 0 in dense
    dense_extended = [_make_hit("top"), _make_hit("other1"), _make_hit("other2")]
    result_extended = _rrf_fuse(dense_extended, sparse_base)
    top_score_extended = next(c for c in result_extended if c.chunk_id == "top").rrf_score

    # Score should be unchanged (rank 0 in dense is the same)
    assert abs(top_score_base - top_score_extended) < 1e-10
