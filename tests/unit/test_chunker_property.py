"""Property-based tests for the chunker using Hypothesis."""

from __future__ import annotations

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from rag.ingestion.chunker import chunk_text


@given(
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=2000,
    ),
    max_tokens=st.integers(min_value=32, max_value=512),
    overlap_tokens=st.integers(min_value=0, max_value=64),
)
@hyp_settings(max_examples=50, deadline=10_000)
def test_chunks_cover_all_input_words(
    text: str, max_tokens: int, overlap_tokens: int
) -> None:
    """For any input text, no words should be lost across all chunks."""
    if not text.strip():
        return

    overlap = min(overlap_tokens, max_tokens // 2)
    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap)

    if not chunks:
        return

    # All words in the input must appear somewhere in the chunks
    input_words = set(text.lower().split())
    chunk_words: set[str] = set()
    for chunk in chunks:
        chunk_words.update(chunk.text.lower().split())

    missing = input_words - chunk_words
    assert not missing, f"Words lost in chunking: {list(missing)[:5]}"


@given(
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=10,
        max_size=1000,
    )
)
@hyp_settings(max_examples=30, deadline=10_000)
def test_chunk_indices_are_sequential_and_zero_based(text: str) -> None:
    """Chunk indices must be sequential starting from 0."""
    if not text.strip():
        return

    chunks = chunk_text(text, max_tokens=128)
    if not chunks:
        return

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@given(
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=5,
        max_size=500,
    )
)
@hyp_settings(max_examples=30, deadline=10_000)
def test_all_chunks_have_positive_token_count(text: str) -> None:
    """Every chunk must have token_count > 0."""
    if not text.strip():
        return

    chunks = chunk_text(text, max_tokens=256)
    for chunk in chunks:
        assert chunk.token_count > 0
