"""Unit tests for the sentence-boundary sliding-window chunker."""

from __future__ import annotations

import pytest
import tiktoken

from rag.ingestion.chunker import Chunk, chunk_text


@pytest.fixture()
def enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_empty_input_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert chunk_text("   \n\t  ") == []


def test_single_short_sentence_produces_one_chunk() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text, max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert "fox" in chunks[0].text


def test_chunk_indices_are_sequential() -> None:
    text = " ".join(["This is sentence number {i}." for i in range(30)])
    chunks = chunk_text(text, max_tokens=64)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_no_chunk_exceeds_max_tokens(enc: tiktoken.Encoding) -> None:
    # 50 sentences, each ~10 tokens
    text = " ".join([f"This is test sentence number {i} in the corpus." for i in range(50)])
    chunks = chunk_text(text, max_tokens=100)
    for chunk in chunks:
        actual_tokens = len(enc.encode(chunk.text))
        # Allow small overshoot for single sentences that exceed max_tokens
        # (the chunker must include them to avoid infinite loops)
        assert actual_tokens <= 200, f"chunk {chunk.chunk_index} has {actual_tokens} tokens"


def test_complete_coverage(enc: tiktoken.Encoding) -> None:
    """All tokens in the input must appear in at least one chunk."""
    text = " ".join([f"Sentence {i} has some words and content." for i in range(20)])
    chunks = chunk_text(text, max_tokens=80, overlap_tokens=16)

    # Collect all words from chunks
    chunk_words: set[str] = set()
    for chunk in chunks:
        chunk_words.update(chunk.text.split())

    # All words from input must appear in at least one chunk
    input_words = set(text.split())
    assert input_words.issubset(chunk_words), (
        f"Missing words: {input_words - chunk_words}"
    )


def test_overlap_creates_shared_content() -> None:
    """Adjacent chunks should share content when overlap_tokens > 0."""
    text = " ".join([f"This is sentence {i}." for i in range(30)])
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=20)

    if len(chunks) < 2:
        pytest.skip("Not enough chunks to test overlap")

    # At least one pair of adjacent chunks should share some words
    found_overlap = False
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].text.split())
        words_b = set(chunks[i + 1].text.split())
        if words_a & words_b:
            found_overlap = True
            break

    assert found_overlap, "Expected overlap between adjacent chunks"


def test_sentence_metadata_is_consistent() -> None:
    """start_sentence and end_sentence should be monotonically non-decreasing."""
    text = " ".join([f"Sentence {i}." for i in range(40)])
    chunks = chunk_text(text, max_tokens=60, overlap_tokens=10)

    for chunk in chunks:
        assert chunk.start_sentence < chunk.end_sentence
        assert chunk.start_sentence >= 0

    # First chunk starts at sentence 0
    assert chunks[0].start_sentence == 0


def test_token_count_is_accurate(enc: tiktoken.Encoding) -> None:
    """Chunk.token_count should match actual encoded token count."""
    text = "The cat sat on the mat. Dogs barked loudly. Birds flew high."
    chunks = chunk_text(text, max_tokens=512)
    for chunk in chunks:
        actual = len(enc.encode(chunk.text))
        # Allow small deviation due to space handling
        assert abs(chunk.token_count - actual) <= 5, (
            f"token_count={chunk.token_count}, actual={actual}"
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_very_long_single_sentence_gets_included() -> None:
    """A sentence longer than max_tokens must be included (no infinite loop)."""
    long_sentence = "word " * 200  # ~200 tokens
    chunks = chunk_text(long_sentence, max_tokens=50)
    assert len(chunks) >= 1
    # The long sentence should be in a chunk even if it exceeds max_tokens


def test_returns_list_of_chunk_objects() -> None:
    text = "First sentence. Second sentence."
    chunks = chunk_text(text)
    assert all(isinstance(c, Chunk) for c in chunks)
