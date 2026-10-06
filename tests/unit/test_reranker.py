"""Unit tests for cross-encoder reranker and MMR diversity re-selection (FR-RAG-14, FR-RAG-15)."""

import pytest

from libs.retrieval.models import SearchResult
from libs.retrieval.reranker import CrossEncoderReranker, compute_token_jaccard


@pytest.mark.unit
def test_token_jaccard_similarity() -> None:
    """Validate token Jaccard similarity between texts."""
    sim_identical = compute_token_jaccard("apple banana orange", "apple banana orange")
    assert sim_identical == 1.0

    sim_disjoint = compute_token_jaccard("apple banana", "car train")
    assert sim_disjoint == 0.0

    sim_partial = compute_token_jaccard("apple banana", "banana orange")
    assert sim_partial == pytest.approx(1 / 3)


@pytest.mark.unit
def test_cross_encoder_scoring_and_reranking() -> None:
    """Validate cross-encoder reranker promotes chunks with highest query alignment (FR-RAG-14)."""
    reranker = CrossEncoderReranker(use_mock=True)

    candidates = [
        SearchResult(
            chunk_id="c1",
            document_id="doc1",
            tenant_id="t1",
            collection_id="col1",
            content="General introduction to web application architecture and cloud components",
            score=0.015,
        ),
        SearchResult(
            chunk_id="c2",
            document_id="doc2",
            tenant_id="t1",
            collection_id="col1",
            content="PostgreSQL database connection pooling with PgBouncer to prevent connection exhaustion",
            score=0.010,
        ),
        SearchResult(
            chunk_id="c3",
            document_id="doc3",
            tenant_id="t1",
            collection_id="col1",
            content="Kubernetes pod autoscaling rules based on CPU and memory thresholds",
            score=0.012,
        ),
    ]

    query = "PostgreSQL connection pooling PgBouncer"
    reranked = reranker.rerank(query, candidates, top_k=3)

    assert len(reranked) == 3
    # c2 has exact match for PostgreSQL connection pooling PgBouncer
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > 0.5


@pytest.mark.unit
def test_reranker_min_score_filtering() -> None:
    """Validate min_score cutoff filters irrelevant candidates."""
    reranker = CrossEncoderReranker(use_mock=True)

    candidates = [
        SearchResult(
            chunk_id="c1",
            document_id="doc1",
            tenant_id="t1",
            collection_id="col1",
            content="Exact match for financial revenue quarterly report metrics",
            score=0.02,
        ),
        SearchResult(
            chunk_id="c2",
            document_id="doc2",
            tenant_id="t1",
            collection_id="col1",
            content="Irrelevant discussion regarding office cafeteria menu items",
            score=0.01,
        ),
    ]

    reranked = reranker.rerank(
        query="financial revenue quarterly report",
        candidates=candidates,
        min_score=0.4,
    )

    assert len(reranked) == 1
    assert reranked[0].chunk_id == "c1"


@pytest.mark.unit
def test_reranker_mmr_diversity() -> None:
    """Validate MMR re-selection penalizes duplicate chunks to ensure diversity (FR-RAG-15)."""
    reranker = CrossEncoderReranker(use_mock=True)

    candidates = [
        SearchResult(
            chunk_id="c1",
            document_id="doc1",
            tenant_id="t1",
            collection_id="col1",
            content="Redis in-memory caching for session storage and cache invalidation",
            score=0.02,
        ),
        SearchResult(
            chunk_id="c2",
            document_id="doc1",
            tenant_id="t1",
            collection_id="col1",
            content="Redis in-memory caching for session storage and cache expiry",
            score=0.019,
        ),
        SearchResult(
            chunk_id="c3",
            document_id="doc2",
            tenant_id="t1",
            collection_id="col1",
            content="Redis distributed replication and high-availability Sentinel clustering",
            score=0.018,
        ),
    ]

    # Without MMR (diversity_weight=None), c1 and c2 (which are near duplicates) both appear at top
    standard = reranker.rerank("Redis caching storage clustering", candidates, top_k=2)
    assert [c.chunk_id for c in standard] == ["c1", "c2"]

    # With MMR (diversity_weight=0.5), c2 is penalized for redundancy with c1, and c3 is chosen
    diverse = reranker.rerank(
        "Redis caching storage clustering",
        candidates,
        top_k=2,
        diversity_weight=0.5,
    )
    assert len(diverse) == 2
    assert diverse[0].chunk_id == "c1"
    assert diverse[1].chunk_id == "c3"
