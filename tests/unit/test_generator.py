"""Unit tests for RAGGenerator end-to-end question answering pipeline (FR-RAG-26, FR-RAG-27, FR-RAG-28)."""

import pytest
from qdrant_client import QdrantClient

from libs.llm.provider import MockLLMProvider
from libs.retrieval.citations import CitationEngine
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.generator import RAGGenerator
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.models import Chunk
from libs.retrieval.reranker import CrossEncoderReranker
from libs.retrieval.sparse import BM25SparseEncoder


@pytest.fixture
def rag_pipeline() -> tuple[QdrantHybridIndexer, RAGGenerator]:
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
    reranker = CrossEncoderReranker(use_mock=True)
    citation_engine = CitationEngine()
    llm_provider = MockLLMProvider(
        canned_response=(
            "Microservices communicate over gRPC and emit OpenTelemetry traces [1]. "
            "High-risk database mutations require HMAC tokens [2]."
        )
    )

    generator = RAGGenerator(
        retriever=retriever,
        reranker=reranker,
        citation_engine=citation_engine,
        llm_provider=llm_provider,
    )
    return indexer, generator


@pytest.mark.unit
def test_rag_generator_produces_grounded_answer(
    rag_pipeline: tuple[QdrantHybridIndexer, RAGGenerator],
) -> None:
    """Validate end-to-end retrieval, reranking, and citation generation."""
    indexer, generator = rag_pipeline
    col_name = "test_generator_col"

    chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000201",
            document_id="doc-arch",
            tenant_id="tenant-acme",
            collection_id="col-docs",
            content="Microservices communicate over gRPC and emit OpenTelemetry traces.",
            page_number=2,
            content_hash="h201",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000202",
            document_id="doc-security",
            tenant_id="tenant-acme",
            collection_id="col-docs",
            content="High-risk database mutations require HMAC tokens approved by admins.",
            page_number=5,
            content_hash="h202",
        ),
    ]
    indexer.index_chunks(col_name, chunks)

    answer = generator.generate(
        query="How do microservices communicate and what tokens are required?",
        tenant_id="tenant-acme",
        collection_id="col-docs",
        collection_name=col_name,
        top_k=2,
    )

    assert answer.is_refusal is False
    assert len(answer.citations) == 2
    assert answer.citations[0].document_id in ["doc-arch", "doc-security"]
    assert answer.confidence_score >= 0.8
    assert "[1]" in answer.answer
    assert "[2]" in answer.answer


@pytest.mark.unit
def test_rag_generator_refusal_on_empty_evidence(
    rag_pipeline: tuple[QdrantHybridIndexer, RAGGenerator],
) -> None:
    """Validate model returns 'I don't know' when context is empty or missing (FR-RAG-28)."""
    _, generator = rag_pipeline

    # Querying a non-existent collection
    answer = generator.generate(
        query="What is the internal revenue of Acme in 2030?",
        tenant_id="tenant-acme",
        collection_id="nonexistent-col",
        collection_name="nonexistent-col",
    )

    assert answer.is_refusal is True
    assert "I don't know based on the provided documents" in answer.answer
    assert len(answer.citations) == 0
    assert answer.confidence_score == 1.0


@pytest.mark.unit
def test_rag_generator_with_cache_compression_and_crag(
    rag_pipeline: tuple[QdrantHybridIndexer, RAGGenerator],
) -> None:
    """Validate integrated caching, CRAG grading, and context compression in generator."""
    from libs.retrieval.cache import SemanticCache
    from libs.retrieval.compressor import ContextCompressor
    from libs.retrieval.crag import AdaptiveCRAGRouter

    indexer, base_generator = rag_pipeline
    col_name = "test_opt_col"

    chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000301",
            document_id="doc-opt",
            tenant_id="tenant-opt",
            collection_id="col-opt",
            content="PostgreSQL high availability is achieved using Patroni with streaming replication.",
            page_number=1,
            content_hash="h301",
        )
    ]
    indexer.index_chunks(col_name, chunks)

    cache = SemanticCache()
    compressor = ContextCompressor()
    crag_router = AdaptiveCRAGRouter()

    generator = RAGGenerator(
        retriever=base_generator.retriever,
        reranker=base_generator.reranker,
        citation_engine=base_generator.citation_engine,
        llm_provider=base_generator.llm_provider,
        cache=cache,
        compressor=compressor,
        crag_router=crag_router,
    )

    # 1. First execution: populates cache
    answer1 = generator.generate(
        query="Explain PostgreSQL high availability streaming replication",
        tenant_id="tenant-opt",
        collection_id="col-opt",
        collection_name=col_name,
    )
    assert answer1.is_refusal is False
    assert len(answer1.citations) > 0

    # 2. Second execution: served directly from Tier 1 cache
    answer2 = generator.generate(
        query="Explain PostgreSQL high availability streaming replication",
        tenant_id="tenant-opt",
        collection_id="col-opt",
        collection_name=col_name,
    )
    assert answer2.answer == answer1.answer

    # 3. Third execution with completely irrelevant query: CRAG routes to refusal
    answer3 = generator.generate(
        query="Quantum gravity string theory supersymmetry",
        tenant_id="tenant-opt",
        collection_id="col-opt",
        collection_name=col_name,
    )
    assert answer3.is_refusal is True
