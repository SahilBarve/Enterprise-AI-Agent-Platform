"""Data models for document parsing, chunking, and retrieval."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChunkStrategy(StrEnum):
    """Supported chunking strategies per collection (FR-RAG-5)."""

    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"
    HEADING_AWARE = "heading_aware"
    TABLE_AWARE = "table_aware"


class DocumentMetadata(BaseModel):
    """Extracted metadata for a document (FR-RAG-6)."""

    tenant_id: str = Field(description="Mandatory tenant scope for isolation")
    collection_id: str = Field(description="Collection identifier")
    title: str = Field(default="Untitled", description="Document title")
    author: str | None = Field(default=None, description="Document author")
    doc_type: str = Field(default="generic", description="File extension or document category")
    created_at: str | None = Field(default=None, description="Document creation or file timestamp")
    language: str = Field(default="en", description="Detected language")
    acl_tags: list[str] = Field(default_factory=list, description="Access control tags")
    source_uri: str | None = Field(default=None, description="S3 or local source location")
    content_hash: str = Field(description="SHA-256 hash of raw document bytes (FR-RAG-7)")
    custom_fields: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary user metadata"
    )


class BlockType(StrEnum):
    """Layout block types identified during document parsing."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    CODE = "code"
    LIST_ITEM = "list_item"


class ParsedBlock(BaseModel):
    """A layout-aware structural block within a parsed document (FR-RAG-4)."""

    block_type: BlockType = Field(default=BlockType.PARAGRAPH)
    content: str = Field(description="Normalized text or markdown table")
    page_number: int = Field(default=1, description="1-indexed document page number")
    section_path: list[str] = Field(default_factory=list, description="Heading hierarchy path")
    heading_level: int | None = Field(default=None, description="1 for H1, 2 for H2, etc.")
    table_headers: list[str] | None = Field(
        default=None, description="Header column names if table"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """A fully ingested and parsed document ready for chunking."""

    document_id: str = Field(description="Unique document ID (e.g. UUID)")
    filename: str = Field(description="Original filename")
    content_type: str = Field(description="MIME type or file extension")
    metadata: DocumentMetadata
    blocks: list[ParsedBlock] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all block contents into a continuous string."""
        return "\n\n".join(b.content for b in self.blocks)


class Chunk(BaseModel):
    """An indexable vector chunk with provenance and parent-child tracking."""

    id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document UUID")
    tenant_id: str = Field(description="Mandatory tenant ID for filtering")
    collection_id: str = Field(description="Target collection ID")
    content: str = Field(description="Chunk text to be embedded")
    chunk_index: int = Field(default=0, description="Sequential order in document")
    page_number: int = Field(default=1, description="Page number for inline citation")
    section_path: list[str] = Field(default_factory=list, description="Section heading breadcrumbs")
    parent_id: str | None = Field(
        default=None,
        description="ID of parent context chunk for small-to-big retrieval (FR-RAG-5, FR-RAG-16)",
    )
    is_parent: bool = Field(
        default=False,
        description="True if this is a large container context chunk",
    )
    token_count: int = Field(default=0, description="Estimated token count")
    content_hash: str = Field(description="SHA-256 hash of chunk content for dedup")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Ranked retrieval candidate with fusion provenance and citation metadata."""

    chunk_id: str
    document_id: str
    tenant_id: str
    collection_id: str
    content: str
    expanded_content: str | None = Field(
        default=None,
        description="Expanded parent chunk content if small-to-big retrieval is applied",
    )
    score: float = Field(description="Final fusion or rerank score")
    dense_rank: int | None = Field(default=None, description="1-indexed rank from dense search")
    sparse_rank: int | None = Field(default=None, description="1-indexed rank from sparse search")
    dense_score: float | None = Field(default=None, description="Raw cosine/similarity score")
    sparse_score: float | None = Field(default=None, description="Raw BM25 score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder relevance score")
    page_number: int = Field(default=1, description="Page number for citation")
    section_path: list[str] = Field(default_factory=list, description="Section breadcrumbs")
    parent_id: str | None = Field(default=None, description="Parent chunk ID if child")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_content(self) -> str:
        """Return expanded content if available, else original chunk content."""
        return self.expanded_content if self.expanded_content is not None else self.content


class RetrievalQuery(BaseModel):
    """Parameters for hybrid retrieval execution."""

    query: str = Field(description="User search or question text")
    tenant_id: str = Field(description="Mandatory tenant isolation identifier")
    collection_id: str = Field(description="Target Qdrant collection")
    top_k: int = Field(default=10, description="Number of results to return after fusion/reranking")
    dense_limit: int = Field(default=40, description="Top candidates retrieved from dense search")
    sparse_limit: int = Field(default=40, description="Top candidates retrieved from sparse search")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing parameter")
    dense_weight: float = Field(default=1.0, description="Weight multiplier for dense RRF score")
    sparse_weight: float = Field(default=1.0, description="Weight multiplier for sparse RRF score")
    expand_parent: bool = Field(
        default=False,
        description="Whether to retrieve and substitute parent context for child chunks",
    )
    filter_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional payload filters",
    )

