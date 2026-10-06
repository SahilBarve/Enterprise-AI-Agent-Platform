"""Unit tests for hybrid retrieval, RRF fusion, tenant isolation, and parent context expansion (FR-RAG-11, FR-RAG-13, FR-RAG-16)."""

import pytest
from qdrant_client import QdrantClient

from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.hybrid import HybridRetriever, calculate_rrf_score
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.models import Chunk, RetrievalQuery
from libs.retrieval.sparse import BM25SparseEncoder


@pytest.mark.unit
def test_calculate_rrf_score() -> None:
    """Validate numerical correctness of Reciprocal Rank Fusion formula."""
    # Rank 1 in both dense and sparse with k=60
    # Score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    score = calculate_rrf_score(dense_rank=1, sparse_rank=1, k=60)
    assert score == pytest.approx(0.032787, rel=1e-3)

    # Present only in dense at rank 2
    # Score = 1/(60+2) = 1/62 ≈ 0.016129
    score_dense_only = calculate_rrf_score(dense_rank=2, sparse_rank=None, k=60)
    assert score_dense_only == pytest.approx(0.016129, rel=1e-3)

    # Custom weights: 2.0 * dense + 1.0 * sparse
    score_weighted = calculate_rrf_score(
        dense_rank=1, sparse_rank=1, k=60, dense_weight=2.0, sparse_weight=1.0
    )
    assert score_weighted == pytest.approx(3 / 61, rel=1e-3)


@pytest.fixture
def hybrid_setup() -> tuple[QdrantClient, QdrantHybridIndexer, HybridRetriever]:
    client = QdrantClient(location=":memory:")
    dense_model = DenseEmbeddingModel(dimension=64, use_mock=True)
    sparse_encoder = BM25SparseEncoder()
    indexer = QdrantHybridIndexer(
        client=client,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
    )
    retriever = HybridRetriever(
        client=client,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
        indexer=indexer,
    )
    return client, indexer, retriever


@pytest.mark.unit
def test_hybrid_retrieval_combines_dense_and_sparse(
    hybrid_setup: tuple[QdrantClient, QdrantHybridIndexer, HybridRetriever],
) -> None:
    """Validate hybrid retrieval fuses dense and sparse hits into ranked results."""
    client, indexer, retriever = hybrid_setup
    col_name = "test_hybrid_fusion"

    chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="doc-postgres",
            tenant_id="tenant-alpha",
            collection_id="col-main",
            content="PostgreSQL relational database handles transactions and ACID compliance",
            chunk_index=0,
            content_hash="h1",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000002",
            document_id="doc-redis",
            tenant_id="tenant-alpha",
            collection_id="col-main",
            content="Redis in-memory cache accelerates fast key-value lookups",
            chunk_index=1,
            content_hash="h2",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000003",
            document_id="doc-rabbit",
            tenant_id="tenant-alpha",
            collection_id="col-main",
            content="RabbitMQ message broker routes async tasks across worker queues",
            chunk_index=2,
            content_hash="h3",
        ),
    ]
    indexer.index_chunks(col_name, chunks)

    query = RetrievalQuery(
        query="relational database transactions",
        tenant_id="tenant-alpha",
        collection_id="col-main",
        top_k=2,
    )

    results = retriever.search(query, collection_name=col_name)

    assert len(results) > 0
    # The postgres document should rank highest due to exact lexical and conceptual match
    top_hit = results[0]
    assert top_hit.chunk_id == "00000000-0000-0000-0000-000000000001"
    assert top_hit.document_id == "doc-postgres"
    assert top_hit.score > 0.0
    assert top_hit.tenant_id == "tenant-alpha"


@pytest.mark.unit
def test_hybrid_retrieval_tenant_isolation(
    hybrid_setup: tuple[QdrantClient, QdrantHybridIndexer, HybridRetriever],
) -> None:
    """Validate strict tenant isolation (FR-GW-3, FR-RAG-13)."""
    client, indexer, retriever = hybrid_setup
    col_name = "test_tenant_isolation"

    chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000010",
            document_id="doc-alpha",
            tenant_id="tenant-alpha",
            collection_id="col-shared",
            content="Alpha confidential financial forecast and revenue numbers",
            content_hash="h10",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000020",
            document_id="doc-beta",
            tenant_id="tenant-beta",
            collection_id="col-shared",
            content="Beta confidential financial forecast and revenue numbers",
            content_hash="h20",
        ),
    ]
    indexer.index_chunks(col_name, chunks)

    # Query from tenant-alpha
    query_alpha = RetrievalQuery(
        query="confidential financial forecast",
        tenant_id="tenant-alpha",
        collection_id="col-shared",
        top_k=5,
    )
    results_alpha = retriever.search(query_alpha, collection_name=col_name)
    assert len(results_alpha) == 1
    assert results_alpha[0].tenant_id == "tenant-alpha"
    assert results_alpha[0].chunk_id == "00000000-0000-0000-0000-000000000010"

    # Query from tenant-beta
    query_beta = RetrievalQuery(
        query="confidential financial forecast",
        tenant_id="tenant-beta",
        collection_id="col-shared",
        top_k=5,
    )
    results_beta = retriever.search(query_beta, collection_name=col_name)
    assert len(results_beta) == 1
    assert results_beta[0].tenant_id == "tenant-beta"
    assert results_beta[0].chunk_id == "00000000-0000-0000-0000-000000000020"


@pytest.mark.unit
def test_hybrid_retrieval_nonexistent_collection(
    hybrid_setup: tuple[QdrantClient, QdrantHybridIndexer, HybridRetriever],
) -> None:
    """Validate graceful handling of unindexed/nonexistent collections."""
    _, _, retriever = hybrid_setup
    query = RetrievalQuery(
        query="test",
        tenant_id="t1",
        collection_id="missing_col",
    )
    results = retriever.search(query, collection_name="missing_col")
    assert results == []


@pytest.mark.unit
def test_parent_context_expansion(
    hybrid_setup: tuple[QdrantClient, QdrantHybridIndexer, HybridRetriever],
) -> None:
    """Validate small-to-big retrieval expands child chunk with parent context (FR-RAG-16)."""
    client, indexer, retriever = hybrid_setup
    col_name = "test_parent_child_expansion"

    parent_id = "00000000-0000-0000-0000-000000000100"
    child_id = "00000000-0000-0000-0000-000000000101"

    parent_content = (
        "Enterprise Architecture Policy Overview: All microservices must implement "
        "health probes, emit OpenTelemetry traces, and register with the service mesh."
    )
    child_content = "emit OpenTelemetry traces"

    chunks = [
        Chunk(
            id=parent_id,
            document_id="doc-arch",
            tenant_id="tenant-acme",
            collection_id="col-policies",
            content=parent_content,
            is_parent=True,
            content_hash="hp",
        ),
        Chunk(
            id=child_id,
            document_id="doc-arch",
            tenant_id="tenant-acme",
            collection_id="col-policies",
            content=child_content,
            parent_id=parent_id,
            is_parent=False,
            content_hash="hc",
        ),
    ]
    indexer.index_chunks(col_name, chunks)

    # 1. Search without expand_parent
    query_no_exp = RetrievalQuery(
        query="OpenTelemetry traces",
        tenant_id="tenant-acme",
        collection_id="col-policies",
        expand_parent=False,
    )
    res_no_exp = retriever.search(query_no_exp, collection_name=col_name)
    assert len(res_no_exp) == 1
    assert res_no_exp[0].chunk_id == child_id
    assert res_no_exp[0].expanded_content is None
    assert res_no_exp[0].effective_content == child_content

    # 2. Search with expand_parent
    query_with_exp = RetrievalQuery(
        query="OpenTelemetry traces",
        tenant_id="tenant-acme",
        collection_id="col-policies",
        expand_parent=True,
    )
    res_with_exp = retriever.search(query_with_exp, collection_name=col_name)
    assert len(res_with_exp) == 1
    assert res_with_exp[0].chunk_id == child_id
    assert res_with_exp[0].expanded_content == parent_content
    assert res_with_exp[0].effective_content == parent_content
