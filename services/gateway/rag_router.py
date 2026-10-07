"""REST API endpoints for Document RAG ingestion, search, and question answering.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why expose RAG via dedicated API Gateway endpoints?
--------------------------------------------------------------------------------
In an enterprise multi-agent platform, agents and external frontends should never
talk directly to vector databases or embedding models. Instead, the API Gateway
acts as an authenticated, observable, and rate-limited boundary:

1. Tenant Isolation Enforcement:
   Every incoming endpoint (`POST /collections`, `POST /documents`, `POST /search`,
   `POST /query`) demands `tenant_id`. The gateway namespaces collection names
   (e.g. `col_{tenant_id}_{collection_id}`) and attaches payload filters, ensuring
   zero cross-tenant data access.

2. Unified Ingestion Lifecycle:
   The `/documents` endpoint chains the entire pipeline:
   `DocumentParser` (extracts blocks & hashes) -> `DocumentChunker` (splits text) ->
   `QdrantHybridIndexer` (embeds dense + sparse vectors & stores payload) ->
   `SemanticCache` (invalidates stale cached queries for this collection).

3. Optimized Query Endpoint with Caching & Compression:
   - `/query`: Complete question answering with two-tier semantic cache (Tier 1 exact,
     Tier 2 semantic embedding match), Adaptive CRAG grading, context compression,
     inline `[N]` citation provenance, and claim groundedness auditing.
================================================================================
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.cache import SemanticCache
from libs.retrieval.chunker import DocumentChunker
from libs.retrieval.citations import GroundedAnswer
from libs.retrieval.generator import RAGGenerator
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.models import ChunkStrategy, RetrievalQuery, SearchResult
from libs.retrieval.parser import DocumentParser
from libs.retrieval.reranker import CrossEncoderReranker
from services.gateway.dependencies import (
    get_indexer,
    get_rag_generator,
    get_reranker,
    get_retriever,
    get_semantic_cache,
)

logger = get_logger("gateway.rag")

router = APIRouter(prefix="/api/v1", tags=["RAG & Knowledge"])


class CreateCollectionRequest(BaseModel):
    """Schema for provisioning a new tenant collection."""

    tenant_id: str = Field(description="Mandatory tenant ID for collection isolation")
    collection_id: str = Field(description="Collection name or identifier (e.g. 'runbooks')")


class CreateCollectionResponse(BaseModel):
    """Result of collection creation or verification."""

    collection_name: str
    status: str


class IngestDocumentRequest(BaseModel):
    """Schema for uploading a document into the RAG vector index."""

    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    title: str = Field(default="Untitled", description="Document title")
    content: str = Field(description="Document text, markdown, or HTML payload")
    content_type: str = Field(
        default="text/plain",
        description="MIME type: text/plain, text/markdown, text/html, text/csv, application/pdf",
    )
    chunk_strategy: ChunkStrategy = Field(
        default=ChunkStrategy.RECURSIVE,
        description="Chunking strategy to apply (recursive, parent_child, heading_aware, table_aware)",
    )
    chunk_size: int = Field(default=500, description="Target chunk character length")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size in characters")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata attributes (tags, department, etc.)"
    )


class IngestDocumentResponse(BaseModel):
    """Outcome of document parsing, chunking, and indexing."""

    document_id: str
    chunks_indexed: int
    content_hash: str


class SearchRequest(BaseModel):
    """Schema for executing hybrid retrieval on an indexed collection."""

    query: str = Field(description="Search or question query string")
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    collection_id: str = Field(description="Target collection identifier")
    top_k: int = Field(default=5, description="Number of results to return")
    expand_parent: bool = Field(
        default=False, description="Expand child chunks to parent context (small-to-big)"
    )
    rerank: bool = Field(default=True, description="Apply cross-encoder reranking and MMR")
    min_score: float | None = Field(default=None, description="Minimum score cutoff threshold")


class QueryRequest(BaseModel):
    """Schema for end-to-end question answering with citations."""

    query: str = Field(description="User question to answer")
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    collection_id: str = Field(description="Target collection identifier")
    top_k: int = Field(default=5, description="Number of chunks retrieved for context")
    min_relevance: float = Field(
        default=0.01, description="Minimum relevance score threshold for refusal guard"
    )
    expand_parent: bool = Field(
        default=True, description="Expand context using parent documents (small-to-big)"
    )
    use_cache: bool = Field(
        default=True, description="Enable two-tier exact and semantic response caching (FR-RAG-18)"
    )
    max_context_tokens: int | None = Field(
        default=1500, description="Maximum token budget for prompt context (FR-RAG-24)"
    )


@router.post(
    "/collections",
    response_model=CreateCollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or ensure a vector collection for a tenant",
)
def create_collection(
    req: CreateCollectionRequest,
    indexer: Annotated[QdrantHybridIndexer, Depends(get_indexer)],
) -> CreateCollectionResponse:
    """Create a Qdrant collection configured with dual dense and sparse vectors.

    Idempotent: If the collection already exists, returns status='already_exists'.
    """
    col_name = indexer.get_collection_name(req.tenant_id, req.collection_id)
    created = indexer.ensure_collection(col_name)
    return CreateCollectionResponse(
        collection_name=col_name,
        status="created" if created else "already_exists",
    )


@router.post(
    "/collections/{collection_id}/documents",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Parse, chunk, and index a document into a collection",
)
def ingest_document(
    collection_id: str,
    req: IngestDocumentRequest,
    indexer: Annotated[QdrantHybridIndexer, Depends(get_indexer)],
    cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
) -> IngestDocumentResponse:
    """Ingest a document through the complete parsing and indexing pipeline:

    1. Parse: Structure raw content into layout blocks and compute SHA-256 hash.
    2. Chunk: Apply selected chunking strategy (e.g. recursive, parent-child).
    3. Index: Generate dense and BM25 sparse vectors and batch upsert into Qdrant.
    4. Invalidate: Automatically clear stale cached queries for this collection (FR-RAG-20).
    """
    col_name = indexer.get_collection_name(req.tenant_id, collection_id)

    # 1. Parse document
    parser = DocumentParser()
    ext_map = {
        "text/markdown": ".md",
        "text/html": ".html",
        "text/csv": ".csv",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }
    ext = ext_map.get(req.content_type, ".txt")
    filename = req.title if req.title.endswith(ext) else f"{req.title}{ext}"

    parsed_doc = parser.parse(
        raw_bytes=req.content.encode("utf-8"),
        filename=filename,
        tenant_id=req.tenant_id,
        collection_id=collection_id,
        custom_fields=req.metadata,
    )

    # 2. Chunk document
    chunker = DocumentChunker(
        strategy=req.chunk_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    chunks = chunker.chunk_document(parsed_doc)

    # 3. Index into Qdrant
    indexed_count = indexer.index_chunks(col_name, chunks)

    # 4. Invalidate collection cache upon new ingestion (FR-RAG-20)
    cache.invalidate_collection(req.tenant_id, collection_id)

    logger.info(
        "Document ingested and indexed",
        tenant_id=req.tenant_id,
        collection_id=collection_id,
        document_id=parsed_doc.document_id,
        chunks=indexed_count,
    )

    return IngestDocumentResponse(
        document_id=parsed_doc.document_id,
        chunks_indexed=indexed_count,
        content_hash=parsed_doc.metadata.content_hash,
    )


@router.delete(
    "/collections/{collection_id}/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete document chunks scoped by tenant (FR-RAG-10)",
)
def delete_document(
    collection_id: str,
    document_id: str,
    tenant_id: Annotated[str, Query(description="Mandatory tenant ID for isolation")],
    indexer: Annotated[QdrantHybridIndexer, Depends(get_indexer)],
    cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
) -> dict[str, str]:
    """Delete all chunks belonging to a document and invalidate the collection cache."""
    col_name = indexer.get_collection_name(tenant_id, collection_id)
    indexer.delete_document(col_name, tenant_id=tenant_id, document_id=document_id)
    cache.invalidate_collection(tenant_id, collection_id)
    return {"status": "deleted", "document_id": document_id}


@router.post(
    "/search",
    response_model=list[SearchResult],
    summary="Execute hybrid retrieval with RRF and cross-encoder reranking (FR-RAG-11, FR-RAG-14)",
)
def search(
    req: SearchRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    reranker: Annotated[CrossEncoderReranker, Depends(get_reranker)],
) -> list[SearchResult]:
    """Retrieve ranked chunks matching the query using dense + sparse hybrid fusion.

    When rerank=True, retrieves double candidates (2 * top_k), scores them with
    the Cross-Encoder, and diversifies results using Maximal Marginal Relevance (MMR).
    """
    query_obj = RetrievalQuery(
        query=req.query,
        tenant_id=req.tenant_id,
        collection_id=req.collection_id,
        top_k=req.top_k * 2 if req.rerank else req.top_k,
        expand_parent=req.expand_parent,
    )
    results = retriever.search(query_obj)

    if req.rerank and results:
        results = reranker.rerank(
            query=req.query,
            candidates=results,
            top_k=req.top_k,
            min_score=req.min_score,
        )

    return results[: req.top_k]


@router.post(
    "/query",
    response_model=GroundedAnswer,
    summary="Execute end-to-end question answering with caching, compression, and citations",
)
def query(
    req: QueryRequest,
    generator: Annotated[RAGGenerator, Depends(get_rag_generator)],
) -> GroundedAnswer:
    """Execute end-to-end optimized RAG question answering:

    1. Checks two-tier semantic cache (Tier 1 exact, Tier 2 semantic similarity).
    2. Retrieves relevant passages using hybrid search and reranking.
    3. Evaluates retrieval via Adaptive CRAG (rewrites if ambiguous, refuses if incorrect).
    4. Compresses context, reorders to mitigate 'lost in the middle', and applies token budgets.
    5. Synthesizes an answer using the LLM with bracketed source citations `[N]`.
    6. Audits claims for factual grounding against the cited context passages.
    """
    return generator.generate(
        query=req.query,
        tenant_id=req.tenant_id,
        collection_id=req.collection_id,
        top_k=req.top_k,
        min_relevance=req.min_relevance,
        expand_parent=req.expand_parent,
        use_cache=req.use_cache,
        max_context_tokens=req.max_context_tokens,
    )


@router.post(
    "/collections/{collection_id}/cache/invalidate",
    status_code=status.HTTP_200_OK,
    summary="Invalidate semantic and exact cache for a collection (FR-RAG-20)",
)
def invalidate_collection_cache(
    collection_id: str,
    tenant_id: Annotated[str, Query(description="Mandatory tenant ID for isolation")],
    cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
) -> dict[str, Any]:
    """Invalidate all cached responses for a specific collection."""
    count = cache.invalidate_collection(tenant_id, collection_id)
    return {"status": "invalidated", "collection_id": collection_id, "entries_cleared": count}


@router.post(
    "/tenants/{tenant_id}/cache/flush",
    status_code=status.HTTP_200_OK,
    summary="Flush all cached responses for a tenant (FR-RAG-21)",
)
def flush_tenant_cache(
    tenant_id: str,
    cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
) -> dict[str, Any]:
    """Admin flush: clear all cached responses across all collections for a tenant."""
    count = cache.flush_tenant(tenant_id)
    return {"status": "flushed", "tenant_id": tenant_id, "entries_cleared": count}
