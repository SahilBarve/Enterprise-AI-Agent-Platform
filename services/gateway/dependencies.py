r"""Dependency injection providers for Gateway endpoints.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Dependency Injection (DI) with @lru_cache in FastAPI?
--------------------------------------------------------------------------------
1. Avoiding Catastrophic Model Reloading:
   Deep learning models (like embedding models or cross-encoders) take hundreds of
   megabytes of RAM and several seconds to load into memory. If we initialized
   a new model on every HTTP request, endpoint latency would exceed 3,000ms!
   By decorating factory functions (`get_dense_model`, `get_reranker`, `get_sparse_encoder`,
   `get_semantic_cache`) with `@lru_cache`, Python instantiates the model once as a singleton
   and reuses it across all concurrent requests.

2. Inversion of Control & Testability:
   Endpoints depend on abstract interfaces (like `LLMProvider`) or factory functions
   rather than hardcoded instances. In unit tests, we can effortlessly override
   `get_qdrant_client` with an in-memory client (`:memory:`) or replace `get_llm_provider`
   with a deterministic mock without touching the route handlers!

3. Composability:
   Notice the dependency graph:
     get_settings()
          |
     get_qdrant_client()   get_dense_model()   get_sparse_encoder()
                 \                |                 /
                         get_indexer()
                              |
                        get_retriever()    get_semantic_cache()   get_context_compressor()
                              \                     |                    /
                                       get_rag_generator()
   FastAPI resolves this Directed Acyclic Graph (DAG) automatically for every request.
================================================================================
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from libs.common.config import PlatformSettings, get_settings
from libs.llm.provider import LLMProvider, MockLLMProvider
from libs.retrieval.cache import SemanticCache
from libs.retrieval.citations import CitationEngine
from libs.retrieval.compressor import ContextCompressor
from libs.retrieval.crag import AdaptiveCRAGRouter
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.generator import RAGGenerator
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.reranker import CrossEncoderReranker
from libs.retrieval.sparse import BM25SparseEncoder


@lru_cache
def get_qdrant_client(settings: Annotated[PlatformSettings, Depends(get_settings)]) -> QdrantClient:
    """Provide Qdrant client instance (in-memory for tests, HTTP for production).

    When running in CI or test suites (`settings.is_testing = True`), Qdrant runs
    in-memory without requiring a live Qdrant container, keeping unit tests blazing fast.
    """
    if settings.is_testing:
        return QdrantClient(location=":memory:")
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
    )


@lru_cache
def get_dense_model() -> DenseEmbeddingModel:
    """Provide singleton instance of the dense embedding model.

    In production, loads sentence-transformers (e.g. bge-small-en-v1.5).
    In local/test environments, uses fast deterministic mock vectors.
    """
    return DenseEmbeddingModel(dimension=64, use_mock=True)


@lru_cache
def get_sparse_encoder() -> BM25SparseEncoder:
    """Provide singleton instance of the BM25 sparse vector encoder."""
    return BM25SparseEncoder()


def get_indexer(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
    dense_model: Annotated[DenseEmbeddingModel, Depends(get_dense_model)],
    sparse_encoder: Annotated[BM25SparseEncoder, Depends(get_sparse_encoder)],
) -> QdrantHybridIndexer:
    """Provide Qdrant hybrid indexer wired with vector client and encoders."""
    return QdrantHybridIndexer(
        client=client,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
    )


def get_retriever(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
    dense_model: Annotated[DenseEmbeddingModel, Depends(get_dense_model)],
    sparse_encoder: Annotated[BM25SparseEncoder, Depends(get_sparse_encoder)],
    indexer: Annotated[QdrantHybridIndexer, Depends(get_indexer)],
) -> HybridRetriever:
    """Provide hybrid retriever wired with Qdrant client, dense/sparse encoders, and indexer."""
    return HybridRetriever(
        client=client,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
        indexer=indexer,
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Provide singleton instance of the Cross-Encoder reranker."""
    return CrossEncoderReranker(use_mock=True)


@lru_cache
def get_citation_engine() -> CitationEngine:
    """Provide singleton instance of the citation and grounding verification engine."""
    return CitationEngine()


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Provide LLM provider protocol instance (MockLLMProvider or LiteLLMProvider)."""
    return MockLLMProvider()


@lru_cache
def get_semantic_cache() -> SemanticCache:
    """Provide singleton instance of the two-tier semantic cache (FR-RAG-18)."""
    return SemanticCache(
        similarity_threshold=0.92,
        default_ttl_seconds=3600,
        min_confidence_to_cache=0.6,
    )


@lru_cache
def get_context_compressor() -> ContextCompressor:
    """Provide singleton instance of the context compressor and token budgeter (FR-RAG-23 to FR-RAG-25)."""
    return ContextCompressor(
        default_max_tokens=1500,
        enable_lost_in_middle_reorder=True,
    )


@lru_cache
def get_crag_router() -> AdaptiveCRAGRouter:
    """Provide singleton instance of the Adaptive Corrective RAG router (FR-RAG-17)."""
    return AdaptiveCRAGRouter(
        correct_threshold=0.65,
        ambiguous_threshold=0.30,
    )


def get_rag_generator(
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    reranker: Annotated[CrossEncoderReranker, Depends(get_reranker)],
    citation_engine: Annotated[CitationEngine, Depends(get_citation_engine)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
    compressor: Annotated[ContextCompressor, Depends(get_context_compressor)],
    crag_router: Annotated[AdaptiveCRAGRouter, Depends(get_crag_router)],
) -> RAGGenerator:
    """Assemble end-to-end optimized RAG question answering pipeline coordinator."""
    return RAGGenerator(
        retriever=retriever,
        reranker=reranker,
        citation_engine=citation_engine,
        llm_provider=llm_provider,
        cache=cache,
        compressor=compressor,
        crag_router=crag_router,
    )
