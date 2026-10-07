"""Unit tests for context compression, token budgeting, and lost in the middle reordering (FR-RAG-23 to FR-RAG-25)."""

import pytest

from libs.retrieval.compressor import ContextCompressor
from libs.retrieval.models import SearchResult


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            tenant_id="tenant-alpha",
            collection_id="runbooks",
            content=f"Important operational instruction step {i} regarding database failover.",
            score=1.0 - (i * 0.1),
            page_number=i,
        )
        for i in range(5)
    ]


def test_split_sentences() -> None:
    compressor = ContextCompressor()
    text = "First sentence here. Second sentence follows! Third question? Fourth one."
    sentences = compressor.split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence follows!"


def test_filter_sentences_prunes_filler() -> None:
    compressor = ContextCompressor(min_sentence_relevance=0.1)
    query = "configure pgbouncer pooling mode"

    content = (
        "PgBouncer pooling mode must be set to transaction. "
        "Copyright 2026 Acme Corp all rights reserved. "
        "The quick brown fox jumps over the lazy dog. "
        "Set max_client_conn to 1000 in pgbouncer.ini configuration."
    )

    pruned, dropped = compressor.filter_sentences(query, content)
    assert dropped >= 1
    assert "transaction" in pruned
    assert "pgbouncer.ini" in pruned
    assert "Copyright" not in pruned


def test_filter_sentences_preserves_short_content() -> None:
    compressor = ContextCompressor()
    short_content = "Brief single sentence note."
    pruned, dropped = compressor.filter_sentences("any query", short_content)
    assert pruned == short_content
    assert dropped == 0


def test_lost_in_the_middle_reordering(sample_search_results: list[SearchResult]) -> None:
    compressor = ContextCompressor(enable_lost_in_middle_reorder=True)

    # Input scores are: [1.0, 0.9, 0.8, 0.7, 0.6] -> chunks 0, 1, 2, 3, 4
    reordered = compressor.reorder_lost_in_middle(sample_search_results)
    assert len(reordered) == 5

    # Rank 1 (chunk-0) must be at index 0 (very start)
    assert reordered[0].chunk_id == "chunk-0"

    # Rank 2 (chunk-1) must be at index -1 (very end)
    assert reordered[-1].chunk_id == "chunk-1"

    # Rank 3 (chunk-2) is at index 1
    assert reordered[1].chunk_id == "chunk-2"

    # Rank 5 (chunk-4) is in the middle (index 2)
    assert reordered[2].chunk_id == "chunk-4"


def test_token_budget_manager_enforces_limit(sample_search_results: list[SearchResult]) -> None:
    # Tight token budget allowing only ~2 chunks
    compressor = ContextCompressor(default_max_tokens=35)
    compressed = compressor.compress(
        query="database failover",
        results=sample_search_results,
    )

    # Should retain fewer than 5 chunks
    assert len(compressed.results) < len(sample_search_results)
    assert compressed.compressed_tokens <= 35
    assert compressed.compression_ratio < 1.0


def test_compress_full_pipeline_metrics(sample_search_results: list[SearchResult]) -> None:
    compressor = ContextCompressor(default_max_tokens=500)
    compressed = compressor.compress(
        query="database failover",
        results=sample_search_results,
    )

    assert compressed.initial_tokens > 0
    assert compressed.compressed_tokens > 0
    assert 0.0 < compressed.compression_ratio <= 1.0
    assert len(compressed.results) == 5


def test_compress_empty_results() -> None:
    compressor = ContextCompressor()
    compressed = compressor.compress(query="query", results=[])
    assert compressed.initial_tokens == 0
    assert compressed.compressed_tokens == 0
    assert len(compressed.results) == 0
