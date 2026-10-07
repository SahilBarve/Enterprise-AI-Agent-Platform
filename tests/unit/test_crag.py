"""Unit tests for Adaptive Corrective RAG (CRAG) router (FR-RAG-12, FR-RAG-17)."""

import pytest

from libs.retrieval.crag import AdaptiveCRAGRouter, CRAGConfidence
from libs.retrieval.models import SearchResult


@pytest.fixture
def crag_router() -> AdaptiveCRAGRouter:
    return AdaptiveCRAGRouter(
        correct_threshold=0.55,
        ambiguous_threshold=0.25,
    )


def test_crag_evaluates_correct_confidence(crag_router: AdaptiveCRAGRouter) -> None:
    query = "configure pgbouncer pooling mode"
    high_quality_results = [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="runbooks",
            content="Configure PgBouncer pooling mode to transaction for optimal web connection throughput.",
            score=0.92,
            rerank_score=0.90,
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="runbooks",
            content="PgBouncer configuration file settings and pooling options.",
            score=0.85,
            rerank_score=0.82,
        ),
    ]

    decision = crag_router.evaluate_retrieval(query, high_quality_results)
    assert decision.confidence == CRAGConfidence.CORRECT
    assert decision.action == "proceed_generation"
    assert decision.score >= 0.55
    assert "Proceeding to generation" in decision.rationale


def test_crag_evaluates_ambiguous_confidence(crag_router: AdaptiveCRAGRouter) -> None:
    query = "database scaling"
    marginal_results = [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="runbooks",
            content="General notes on scaling infrastructure services.",
            score=0.45,
            rerank_score=0.40,
        ),
    ]

    decision = crag_router.evaluate_retrieval(query, marginal_results)
    assert decision.confidence == CRAGConfidence.AMBIGUOUS
    assert decision.action == "rewrite_and_retry"
    assert 0.25 <= decision.score < 0.55
    assert decision.rewritten_query is not None


def test_crag_evaluates_incorrect_confidence(crag_router: AdaptiveCRAGRouter) -> None:
    query = "Kafka broker partition replication rebalance"
    irrelevant_results = [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="runbooks",
            content="Frontend CSS styling guidelines and color palettes.",
            score=0.05,
            rerank_score=0.02,
        ),
    ]

    decision = crag_router.evaluate_retrieval(query, irrelevant_results)
    assert decision.confidence == CRAGConfidence.INCORRECT
    assert decision.action == "fallback_external"
    assert decision.score < 0.30
    assert "Triggering fallback" in decision.rationale


def test_crag_handles_empty_candidates(crag_router: AdaptiveCRAGRouter) -> None:
    decision = crag_router.evaluate_retrieval("Any query", [])
    assert decision.confidence == CRAGConfidence.INCORRECT
    assert decision.action == "fallback_external"
    assert decision.score == 0.0


def test_crag_query_rewriting(crag_router: AdaptiveCRAGRouter) -> None:
    raw_query = "Can you please explain how to configure PgBouncer pooling?"
    rewritten = crag_router.rewrite_query(raw_query)
    assert "Can you please" not in rewritten
    assert "explain how to" not in rewritten
    assert "configure PgBouncer pooling" in rewritten


def test_crag_query_decomposition(crag_router: AdaptiveCRAGRouter) -> None:
    complex_query = "Explain Patroni failover steps and configure PgBouncer connection limits"
    sub_queries = crag_router.decompose_query(complex_query)
    assert len(sub_queries) == 2
    assert "Explain Patroni failover steps" in sub_queries[0]
    assert "configure PgBouncer connection limits" in sub_queries[1]
