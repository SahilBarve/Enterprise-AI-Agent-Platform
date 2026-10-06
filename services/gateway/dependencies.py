"""Dependency injection providers for Gateway endpoints."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from libs.common.config import PlatformSettings, get_settings
from libs.llm.provider import LLMProvider, MockLLMProvider
from libs.retrieval.citations import CitationEngine
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.generator import RAGGenerator
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.reranker import CrossEncoderReranker
from libs.retrieval.sparse import BM25SparseEncoder


@lru_cache
def get_qdrant_client(settings: Annotated[PlatformSettings, Depends(get_settings)]) -> QdrantClient:
    """Provide Qdrant client instance (in-memory for tests, HTTP for production)."""
    if settings.is_testing:
        return QdrantClient(location=":memory:")
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
    )


@lru_cache
def get_dense_model() -> DenseEmbeddingModel:
    return DenseEmbeddingModel(dimension=64, use_mock=True)


@lru_cache
def get_sparse_encoder() -> BM25SparseEncoder:
    return BM25SparseEncoder()


def get_indexer(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
    dense_model: Annotated[DenseEmbeddingModel, Depends(get_dense_model)],
    sparse_encoder: Annotated[BM25SparseEncoder, Depends(get_sparse_encoder)],
) -> QdrantHybridIndexer:
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
    return HybridRetriever(
        client=client,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
        indexer=indexer,
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(use_mock=True)


@lru_cache
def get_citation_engine() -> CitationEngine:
    return CitationEngine()


@lru_cache
def get_llm_provider() -> LLMProvider:
    return MockLLMProvider()


def get_rag_generator(
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    reranker: Annotated[CrossEncoderReranker, Depends(get_reranker)],
    citation_engine: Annotated[CitationEngine, Depends(get_citation_engine)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> RAGGenerator:
    return RAGGenerator(
        retriever=retriever,
        reranker=reranker,
        citation_engine=citation_engine,
        llm_provider=llm_provider,
    )
