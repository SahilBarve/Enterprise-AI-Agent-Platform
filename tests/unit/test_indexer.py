"""Unit tests for Qdrant hybrid indexer and collection manager (FR-RAG-10, FR-RAG-11, FR-RAG-13)."""

import pytest
from qdrant_client import QdrantClient

from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantHybridIndexer
from libs.retrieval.models import Chunk
from libs.retrieval.sparse import BM25SparseEncoder


@pytest.fixture
def in_memory_client() -> QdrantClient:
    return QdrantClient(location=":memory:")


@pytest.fixture
def indexer(in_memory_client: QdrantClient) -> QdrantHybridIndexer:
    dense = DenseEmbeddingModel(dimension=64, use_mock=True)
    sparse = BM25SparseEncoder()
    return QdrantHybridIndexer(
        client=in_memory_client,
        dense_model=dense,
        sparse_encoder=sparse,
    )


@pytest.mark.unit
def test_ensure_collection_creates_hybrid_schema(
    indexer: QdrantHybridIndexer, in_memory_client: QdrantClient
) -> None:
    """Validate collection is created with both dense and sparse vector configurations."""
    col_name = "test_col"
    created = indexer.ensure_collection(collection_name=col_name, dense_dim=64)
    assert created is True

    # Check collection info
    info = in_memory_client.get_collection(col_name)
    assert info is not None
    vectors_config = info.config.params.vectors
    assert isinstance(vectors_config, dict)
    assert DENSE_VECTOR_NAME in vectors_config

    sparse_config = info.config.params.sparse_vectors
    assert sparse_config is not None
    assert SPARSE_VECTOR_NAME in sparse_config

    # Second call should not recreate
    assert indexer.ensure_collection(collection_name=col_name, dense_dim=64) is False


@pytest.mark.unit
def test_index_and_delete_chunks(
    indexer: QdrantHybridIndexer, in_memory_client: QdrantClient
) -> None:
    """Validate indexing chunks and cascading deletion by document ID."""
    col_name = "test_indexing"
    chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="col-1",
            content="FastAPI gateway routes requests",
            chunk_index=0,
            content_hash="hash-1",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000002",
            document_id="doc-1",
            tenant_id="tenant-alpha",
            collection_id="col-1",
            content="Qdrant stores dense and sparse vectors",
            chunk_index=1,
            content_hash="hash-2",
        ),
        Chunk(
            id="00000000-0000-0000-0000-000000000003",
            document_id="doc-2",
            tenant_id="tenant-alpha",
            collection_id="col-1",
            content="Separate document chunk",
            chunk_index=0,
            content_hash="hash-3",
        ),
    ]

    indexed_count = indexer.index_chunks(col_name, chunks)
    assert indexed_count == 3

    # Verify point count
    info = in_memory_client.get_collection(col_name)
    assert info.points_count == 3

    # Delete doc-1
    indexer.delete_document(col_name, tenant_id="tenant-alpha", document_id="doc-1")
    info_after = in_memory_client.get_collection(col_name)
    assert info_after.points_count == 1
