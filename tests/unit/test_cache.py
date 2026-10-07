"""Unit tests for two-tier semantic cache (FR-RAG-18 to FR-RAG-22)."""

import time

import pytest

from libs.retrieval.cache import SemanticCache
from libs.retrieval.citations import Citation, GroundedAnswer
from libs.retrieval.embeddings import DenseEmbeddingModel


@pytest.fixture
def sample_grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="PgBouncer should be set to transaction pooling mode [1].",
        citations=[
            Citation(
                source_number=1,
                chunk_id="chunk-101",
                document_id="doc-postgres",
                page_number=2,
                section_path=["Postgres", "PgBouncer"],
                excerpt="Transaction pooling is recommended for web workloads.",
            )
        ],
        confidence_score=0.95,
        is_refusal=False,
    )


@pytest.fixture
def cache() -> SemanticCache:
    embedding_model = DenseEmbeddingModel(dimension=64, use_mock=True)
    return SemanticCache(
        embedding_model=embedding_model,
        similarity_threshold=0.90,
        default_ttl_seconds=3600,
        min_confidence_to_cache=0.6,
    )


def test_exact_cache_hit_tier_1(cache: SemanticCache, sample_grounded_answer: GroundedAnswer) -> None:
    # 1. Miss initially
    miss = cache.get(
        query="How to configure PgBouncer pooling?",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert miss is None

    # 2. Store in cache
    success = cache.set(
        query="How to configure PgBouncer pooling?",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert success is True

    # 3. Retrieve with exact same query (case-insensitive and trimmed)
    hit = cache.get(
        query="  how to configure pgbouncer pooling?  ",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert hit is not None
    assert hit.cache_tier == "exact"
    assert hit.similarity == 1.0
    assert "PgBouncer should be set" in hit.answer.answer


def test_semantic_cache_hit_tier_2(cache: SemanticCache, sample_grounded_answer: GroundedAnswer) -> None:
    # Seed cache with initial query
    cache.set(
        query="How to configure PgBouncer pooling?",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )

    # In mock embedding model, similar words produce high cosine similarity
    # Query with identical words in different casing/spacing
    hit = cache.get(
        query="how to configure pgbouncer pooling",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert hit is not None
    assert hit.cache_tier in ("exact", "semantic")
    assert hit.answer.confidence_score == 0.95


def test_cache_miss_on_dissimilar_query(cache: SemanticCache, sample_grounded_answer: GroundedAnswer) -> None:
    cache.set(
        query="How to configure PgBouncer pooling?",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )

    # Query about completely unrelated topic
    hit = cache.get(
        query="Kubernetes pod autoscaling HPA thresholds",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert hit is None


def test_multi_tenant_isolation(cache: SemanticCache, sample_grounded_answer: GroundedAnswer) -> None:
    # Seed tenant-alpha
    cache.set(
        query="Database failover policy",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )

    # Tenant-beta queries identical question -> must MISS
    hit = cache.get(
        query="Database failover policy",
        tenant_id="tenant-beta",
        collection_id="runbooks",
    )
    assert hit is None


def test_model_and_prompt_version_isolation(
    cache: SemanticCache, sample_grounded_answer: GroundedAnswer
) -> None:
    cache.set(
        query="Database failover policy",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
        model="gpt-4o",
        prompt_version="v1",
    )

    # Same query but different prompt version -> must MISS
    hit_diff_prompt = cache.get(
        query="Database failover policy",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
        model="gpt-4o",
        prompt_version="v2",
    )
    assert hit_diff_prompt is None

    # Same query but different model -> must MISS
    hit_diff_model = cache.get(
        query="Database failover policy",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
        model="claude-3-5-sonnet",
        prompt_version="v1",
    )
    assert hit_diff_model is None


def test_cache_poisoning_prevention_refusal(cache: SemanticCache) -> None:
    refusal_answer = GroundedAnswer(
        answer="I don't know based on the provided documents.",
        citations=[],
        confidence_score=1.0,
        is_refusal=True,
    )

    # Attempt to cache a refusal response
    stored = cache.set(
        query="Unknown query that failed retrieval",
        answer=refusal_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert stored is False

    # Verify not in cache
    hit = cache.get(
        query="Unknown query that failed retrieval",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert hit is None


def test_cache_poisoning_prevention_low_confidence(cache: SemanticCache) -> None:
    low_conf_answer = GroundedAnswer(
        answer="Possibly something about database [1].",
        citations=[],
        confidence_score=0.3,  # Below min_confidence_to_cache=0.6
        is_refusal=False,
    )

    stored = cache.set(
        query="Vague query",
        answer=low_conf_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert stored is False

    hit = cache.get(
        query="Vague query",
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    assert hit is None


def test_ttl_expiration(sample_grounded_answer: GroundedAnswer) -> None:
    # Cache with 1-second TTL
    cache = SemanticCache(default_ttl_seconds=1)
    cache.set(
        query="Expiring query",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
        ttl_seconds=1,
    )

    # Immediate lookup hits
    assert cache.get("Expiring query", "tenant-alpha", "runbooks") is not None

    # Sleep past expiration
    time.sleep(1.1)

    # Lookup misses and expires
    assert cache.get("Expiring query", "tenant-alpha", "runbooks") is None


def test_collection_invalidation_and_tenant_flush(
    cache: SemanticCache, sample_grounded_answer: GroundedAnswer
) -> None:
    cache.set(
        query="Query 1",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="runbooks",
    )
    cache.set(
        query="Query 2",
        answer=sample_grounded_answer,
        tenant_id="tenant-alpha",
        collection_id="policies",
    )

    # Invalidate collection 'runbooks' only
    removed = cache.invalidate_collection(tenant_id="tenant-alpha", collection_id="runbooks")
    assert removed >= 1

    assert cache.get("Query 1", "tenant-alpha", "runbooks") is None
    assert cache.get("Query 2", "tenant-alpha", "policies") is not None

    # Flush tenant completely
    flushed = cache.flush_tenant(tenant_id="tenant-alpha")
    assert flushed >= 1
    assert cache.get("Query 2", "tenant-alpha", "policies") is None
