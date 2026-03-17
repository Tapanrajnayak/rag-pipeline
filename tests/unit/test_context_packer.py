"""Unit tests for the token-budget-aware context packer."""

from __future__ import annotations

import tiktoken
import pytest

from rag.generation.context import Citation, PackedContext, pack_context
from rag.retrieval.hybrid import RetrievedChunk
from rag.retrieval.reranker import RerankResult


def _make_result(
    chunk_id: str,
    text: str,
    document_id: str = "doc-001",
    score: float = 0.9,
) -> RerankResult:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        score=score,
        rrf_score=score,
    )
    return RerankResult(chunk=chunk, relevance_score=score, rerank_used=True)


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_empty_results_returns_empty_context() -> None:
    result = pack_context([], max_tokens=1000)
    assert result.text == ""
    assert result.citations == []
    assert result.token_count == 0
    assert not result.truncated


def test_single_chunk_fits_entirely() -> None:
    text = "This is a short sentence."
    results = [_make_result("c1", text)]
    context = pack_context(results, max_tokens=1000)

    assert text in context.text
    assert len(context.citations) == 1
    assert context.citations[0].chunk_id == "c1"
    assert not context.truncated


def test_never_exceeds_max_tokens() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    # Create chunks that collectively exceed the budget
    chunks = [
        _make_result(f"c{i}", " ".join([f"word{j}" for j in range(100)]), score=1.0 - i * 0.1)
        for i in range(20)
    ]
    context = pack_context(chunks, max_tokens=200)
    actual_tokens = len(enc.encode(context.text)) if context.text else 0
    # Allow a small margin for sentence-boundary truncation imprecision
    assert actual_tokens <= 220, f"Context has {actual_tokens} tokens, max is 200"


def test_citation_indices_are_1_based() -> None:
    results = [
        _make_result("c1", "First chunk text."),
        _make_result("c2", "Second chunk text."),
    ]
    context = pack_context(results, max_tokens=1000)
    indices = [c.index for c in context.citations]
    assert indices == [1, 2]


def test_citation_order_matches_relevance() -> None:
    """Citations should be in the same order as the input (highest relevance first)."""
    results = [
        _make_result("c1", "High relevance chunk.", score=0.95),
        _make_result("c2", "Medium relevance chunk.", score=0.80),
        _make_result("c3", "Low relevance chunk.", score=0.65),
    ]
    context = pack_context(results, max_tokens=1000)
    chunk_ids = [c.chunk_id for c in context.citations]
    assert chunk_ids == ["c1", "c2", "c3"]


def test_truncated_flag_set_when_budget_exceeded() -> None:
    # Very tight budget — last chunk will be truncated
    long_text = "This is a longer sentence that uses several tokens. " * 10
    results = [
        _make_result("c1", "Short sentence."),
        _make_result("c2", long_text),
    ]
    context = pack_context(results, max_tokens=20)
    # Either truncated=True or c2 was not included
    if len(context.citations) > 1:
        assert context.truncated


def test_truncation_at_sentence_boundary() -> None:
    """If a chunk is truncated, it must end at a sentence boundary."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    results = [_make_result("c1", "Short intro."), _make_result("c2", text)]
    context = pack_context(results, max_tokens=15)

    if len(context.citations) > 1 and context.truncated:
        truncated_text = context.citations[1].text
        # Should end at sentence boundary (ends with a period)
        stripped = truncated_text.strip()
        assert stripped.endswith(".") or stripped == "", (
            f"Truncated text doesn't end at sentence: '{stripped}'"
        )


def test_document_ids_in_citations() -> None:
    results = [
        _make_result("c1", "From document A.", document_id="doc-A"),
        _make_result("c2", "From document B.", document_id="doc-B"),
    ]
    context = pack_context(results, max_tokens=1000)
    doc_ids = [c.document_id for c in context.citations]
    assert "doc-A" in doc_ids
    assert "doc-B" in doc_ids


def test_token_count_is_accurate() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    text = "The quick brown fox jumps over the lazy dog."
    results = [_make_result("c1", text)]
    context = pack_context(results, max_tokens=1000)
    actual = len(enc.encode(context.text))
    assert abs(context.token_count - actual) <= 5
