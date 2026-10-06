"""Qdrant hybrid vector indexer and collection manager (FR-RAG-3, FR-RAG-10, FR-RAG-11, FR-RAG-13)."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from libs.common.logging import get_logger
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.models import Chunk
from libs.retrieval.sparse import BM25SparseEncoder

logger = get_logger("retrieval.indexer")

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
        self.client = client
        self.dense_model = dense_model or DenseEmbeddingModel(use_mock=True)
        self.sparse_encoder = sparse_encoder or BM25SparseEncoder()

    def get_collection_name(self, tenant_id: str, collection_id: str) -> str:
        """Derive isolated collection name or namespace key."""
        return f"col_{tenant_id}_{collection_id}".replace("-", "_")

    def ensure_collection(
        self,
        collection_name: str,
        dense_dim: int = 384,
        distance: rest.Distance = rest.Distance.COSINE,
    ) -> bool:
        """Create hybrid collection with named dense and sparse vectors if not existing."""
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
        """Embed and upsert chunks into Qdrant with dual vectors and payload metadata."""
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
        """Cascade delete all chunks belonging to a document under tenant scope (FR-RAG-10)."""
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
