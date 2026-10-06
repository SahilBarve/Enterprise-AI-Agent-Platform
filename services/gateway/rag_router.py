"""REST API endpoints for Document RAG ingestion, search, and question answering (FR-RAG-1, FR-RAG-10, FR-RAG-11, FR-RAG-26)."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from libs.common.logging import get_logger
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
)

logger = get_logger("gateway.rag")

router = APIRouter(prefix="/api/v1", tags=["RAG & Knowledge"])


class CreateCollectionRequest(BaseModel):
    tenant_id: str = Field(description="Mandatory tenant ID for collection isolation")
    collection_id: str = Field(description="Collection name or identifier")


class CreateCollectionResponse(BaseModel):
    collection_name: str
    status: str


class IngestDocumentRequest(BaseModel):
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    title: str = Field(default="Untitled", description="Document title")
    content: str = Field(description="Document text or markdown payload")
    content_type: str = Field(default="text/plain", description="MIME type: text/plain, text/markdown, text/html, text/csv")
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.RECURSIVE, description="Chunking strategy to apply")
    chunk_size: int = Field(default=500, description="Target chunk character length")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class IngestDocumentResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    content_hash: str


class SearchRequest(BaseModel):
    query: str = Field(description="Search or question query string")
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    collection_id: str = Field(description="Target collection identifier")
    top_k: int = Field(default=5, description="Number of results to return")
    expand_parent: bool = Field(default=False, description="Expand child chunks to parent context")
    rerank: bool = Field(default=True, description="Apply cross-encoder reranker")
    min_score: float | None = Field(default=None, description="Minimum score cutoff")


class QueryRequest(BaseModel):
    query: str = Field(description="User question to answer")
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    collection_id: str = Field(description="Target collection identifier")
    top_k: int = Field(default=5, description="Number of chunks retrieved for context")
    min_relevance: float = Field(default=0.01, description="Minimum relevance score threshold")
    expand_parent: bool = Field(default=True, description="Expand context using parent documents")


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
) -> IngestDocumentResponse:
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
    tenant_id: Annotated[str, Query(description="Mandatory tenant ID")],
    indexer: Annotated[QdrantHybridIndexer, Depends(get_indexer)],
) -> dict[str, str]:
    col_name = indexer.get_collection_name(tenant_id, collection_id)
    indexer.delete_document(col_name, tenant_id=tenant_id, document_id=document_id)
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
    summary="Execute end-to-end question answering with inline citations and groundedness check (FR-RAG-26, FR-RAG-27, FR-RAG-28)",
)
def query(
    req: QueryRequest,
    generator: Annotated[RAGGenerator, Depends(get_rag_generator)],
) -> GroundedAnswer:
    return generator.generate(
        query=req.query,
        tenant_id=req.tenant_id,
        collection_id=req.collection_id,
        top_k=req.top_k,
        min_relevance=req.min_relevance,
        expand_parent=req.expand_parent,
    )
