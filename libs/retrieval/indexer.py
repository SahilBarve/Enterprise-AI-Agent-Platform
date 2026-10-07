"""Qdrant hybrid vector indexer and collection manager (FR-RAG-3, FR-RAG-10, FR-RAG-11, FR-RAG-13).

This module manages the lifecycle of vector collections in Qdrant.
Instead of storing vectors in separate databases, Qdrant allows dual named vectors
in a single point:
  - "dense": 384-dimensional float vector scored via Cosine similarity.
  - "sparse": Inverted index vector of hashed term IDs and BM25 weights.

It also manages payload indexes (keyword indexes on tenant_id, collection_id, etc.)
so queries can filter out unauthorized data before vector distance calculations begin.
"""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from libs.common.logging import get_logger
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.models import Chunk
from libs.retrieval.sparse import BM25SparseEncoder

logger = get_logger("retrieval.indexer")

# Standard vector names used across all collections in the platform
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantHybridIndexer:
    """Manages Qdrant collections with dual dense + sparse vector indexing."""

    def __init__(
        self,
        client: QdrantClient,
        dense_model: DenseEmbeddingModel | None = None,
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> None:
        """Initialize hybrid indexer.

        Args:
            client: Connected QdrantClient instance (HTTP or in-memory).
            dense_model: Model used to generate dense vectors (default: 384-dim mock/fastembed).
            sparse_encoder: BM25 encoder used to generate sparse vectors.
        """
        self.client = client
        self.dense_model = dense_model or DenseEmbeddingModel(use_mock=True)
        self.sparse_encoder = sparse_encoder or BM25SparseEncoder()

    def get_collection_name(self, tenant_id: str, collection_id: str) -> str:
        """Derive isolated collection name or namespace key.

        Qdrant collection names cannot contain hyphens or spaces in certain modes,
        so we normalize to 'col_{tenant_id}_{collection_id}'.

        Args:
            tenant_id: Unique organization or tenant identifier.
            collection_id: Knowledge base or category identifier.

        Returns:
            Normalized Qdrant collection name string.
        """
        return f"col_{tenant_id}_{collection_id}".replace("-", "_")

    def ensure_collection(
        self,
        collection_name: str,
        dense_dim: int = 384,
        distance: rest.Distance = rest.Distance.COSINE,
    ) -> bool:
        """Create hybrid collection with named dense and sparse vectors if not existing.

        Steps:
        1. Checks if collection already exists in Qdrant.
        2. If not, creates collection configured with:
           - vectors_config: named vector 'dense' (size=dense_dim, distance=COSINE).
           - sparse_vectors_config: named vector 'sparse' (on_disk=False for fast RAM search).
        3. Creates payload keyword indexes on:
           - 'tenant_id': mandatory for strict multi-tenant isolation.
           - 'collection_id': sub-scoping within tenant.
           - 'document_id': fast cascading deletes and document updates.
           - 'is_parent': distinguishes parent context chunks from searchable child chunks.

        Args:
            collection_name: Target collection name.
            dense_dim: Dense vector dimensionality (default: 384).
            distance: Distance metric (Cosine, Dot, or Euclid).

        Returns:
            True if collection was created, False if it already existed.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: rest.VectorParams(
                        size=dense_dim,
                        distance=distance,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                        index=rest.SparseIndexParams(on_disk=False)
                    )
                },
            )

            # Create payload indexes for pre-retrieval filtering (FR-RAG-13)
            for field_name in ["tenant_id", "collection_id", "document_id", "is_parent"]:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )

            logger.info("Created hybrid Qdrant collection", collection=collection_name)
            return True
        return False

    def index_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert chunks into Qdrant with dual vectors and payload metadata.

        How it works:
        1. Ensures the target collection exists.
        2. Computes dense vectors for all chunk texts in batch.
        3. Computes sparse BM25 vectors for all chunk texts in batch.
        4. Packages each chunk into a PointStruct containing:
           - Point ID: chunk.id (UUID string).
           - Named vectors: {"dense": [...], "sparse": SparseVector(...)}.
           - Payload: complete chunk text, document ID, page number, section breadcrumbs, etc.
        5. Batch upserts points into Qdrant in chunks of `batch_size`.

        Args:
            collection_name: Target Qdrant collection name.
            chunks: List of Chunk objects to index.
            batch_size: Batch size for network upsert requests (default: 64).

        Returns:
            Number of successfully indexed chunks.
        """
        if not chunks:
            return 0

        self.ensure_collection(collection_name, dense_dim=self.dense_model.dimension)

        points: list[rest.PointStruct] = []
        texts = [c.content for c in chunks]

        # Batch encode dense and sparse vectors
        dense_vectors = self.dense_model.embed_batch(texts)
        sparse_vectors = self.sparse_encoder.encode_batch(texts)

        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "collection_id": chunk.collection_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_path": chunk.section_path,
                "parent_id": chunk.parent_id,
                "is_parent": chunk.is_parent,
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash,
                "metadata": chunk.metadata,
            }

            points.append(
                rest.PointStruct(
                    id=chunk.id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )

        # Batch upsert points
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

        logger.info(
            "Indexed chunks into collection",
            collection=collection_name,
            count=len(chunks),
        )
        return len(chunks)

    def delete_document(
        self,
        collection_name: str,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Cascade delete all chunks belonging to a document under tenant scope (FR-RAG-10).

        Critical Security Design:
        The deletion filter MUST require BOTH tenant_id AND document_id.
        This guarantees that even if a caller passes an arbitrary document_id,
        it can never delete chunks belonging to another tenant.

        Args:
            collection_name: Target collection name.
            tenant_id: Mandatory tenant isolation ID.
            document_id: Unique document UUID whose chunks should be wiped.
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="tenant_id",
                        match=rest.MatchValue(value=tenant_id),
                    ),
                    rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=document_id),
                    ),
                ]
            ),
            wait=True,
        )
        logger.info(
            "Deleted document chunks",
            collection=collection_name,
            document_id=document_id,
            tenant_id=tenant_id,
        )
