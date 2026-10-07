"""Data models for document parsing, chunking, and retrieval.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why do we need structured domain models between parsing and vector storage?
--------------------------------------------------------------------------------
In modern enterprise RAG pipelines, passing raw strings or loose dictionaries
creates brittle code and security hazards:
1. Multi-Tenant Isolation: If `tenant_id` is an optional afterthought, a developer
   could easily write a vector search query that misses the tenant filter, leading
   to critical cross-tenant data leakage. By making `tenant_id` a mandatory Field
   in DocumentMetadata, Chunk, SearchResult, and RetrievalQuery, tenant boundaries
   are enforced by the type system at compile and runtime.
2. Small-to-Big Retrieval (Parent-Child):
   - Small chunks (e.g., 500 chars) produce precise vector embeddings because the
     meaning is concentrated without dilution.
   - Large parent chunks (e.g., 2000 chars) provide complete context to the LLM so
     it does not hallucinate due to truncated sentences or missing paragraphs.
   The `parent_id` and `is_parent` fields in `Chunk` model this explicit hierarchy.
3. Provenance & Citations:
   Every answer produced by an enterprise AI must be auditable. Storing `document_id`,
   `page_number`, `section_path`, and `content_hash` enables inline citations ([1], [2])
   and claims verification back to the source text.
================================================================================
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChunkStrategy(StrEnum):
    """Supported chunking strategies per collection (FR-RAG-5).

    Different document structures require different splitting logic:
    - RECURSIVE: General-purpose sliding window splitting on paragraphs, then sentences.
    - PARENT_CHILD: Creates small search chunks linked to large parent context chunks.
    - HEADING_AWARE: Groups text strictly under common markdown/PDF heading trees.
    - TABLE_AWARE: Keeps tabular data intact and repeats headers across row splits.
    """

    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"
    HEADING_AWARE = "heading_aware"
    TABLE_AWARE = "table_aware"


class DocumentMetadata(BaseModel):
    """Extracted metadata for a document (FR-RAG-6).

    This metadata stays attached to the document record and is propagated down
    to all chunks extracted from this document.
    """

    tenant_id: str = Field(description="Mandatory tenant scope for strict multi-tenant isolation")
    collection_id: str = Field(description="Collection identifier (e.g. 'runbooks', 'policies')")
    title: str = Field(default="Untitled", description="Document title extracted from filename or tags")
    author: str | None = Field(default=None, description="Document author if present in file metadata")
    doc_type: str = Field(default="generic", description="File extension or category (pdf, md, txt, csv)")
    created_at: str | None = Field(default=None, description="Document creation or file timestamp")
    language: str = Field(default="en", description="Detected language for language-specific tokenization")
    acl_tags: list[str] = Field(default_factory=list, description="Access control tags for role-based retrieval")
    source_uri: str | None = Field(default=None, description="S3 or local storage URI for original asset")
    content_hash: str = Field(description="SHA-256 hash of raw document bytes for deduplication (FR-RAG-7)")
    custom_fields: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary tenant-defined user metadata"
    )


class BlockType(StrEnum):
    """Layout block types identified during document parsing (FR-RAG-4).

    Layout-aware parsing recognizes structural boundaries before chunking begins,
    preventing tables or code blocks from being mangled by naive string splitting.
    """

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    CODE = "code"
    LIST_ITEM = "list_item"


class ParsedBlock(BaseModel):
    """A layout-aware structural block within a parsed document (FR-RAG-4).

    Attributes:
        block_type: Structural role of this block (paragraph, heading, table, etc.).
        content: Clean normalized text or markdown representation.
        page_number: 1-indexed document page number (critical for citation provenance).
        section_path: Heading hierarchy breadcrumbs leading to this block
                      (e.g. ["Deployment", "Production", "Rollback"]).
        heading_level: Markdown heading level (1 for H1, 2 for H2, etc.).
        table_headers: Column names if this block is a table.
        metadata: Extra block-specific attributes.
    """

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
    """A fully ingested and parsed document ready for chunking.

    Serves as the intermediate representation (IR) between file parsers and chunking engines.
    """

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
    """An indexable vector chunk with provenance and parent-child tracking.

    Each chunk is an individual unit stored in Qdrant (or vector DB). It contains
    both dense and sparse vector representations plus a rich payload for filtering.
    """

    id: str = Field(description="Unique chunk identifier (UUID)")
    document_id: str = Field(description="Parent document UUID")
    tenant_id: str = Field(description="Mandatory tenant ID for filtering")
    collection_id: str = Field(description="Target collection ID")
    content: str = Field(description="Chunk text to be embedded and searched")
    chunk_index: int = Field(default=0, description="Sequential order within original document")
    page_number: int = Field(default=1, description="Page number for inline citation provenance")
    section_path: list[str] = Field(default_factory=list, description="Section heading breadcrumbs")
    parent_id: str | None = Field(
        default=None,
        description="ID of parent context chunk for small-to-big retrieval (FR-RAG-5, FR-RAG-16)",
    )
    is_parent: bool = Field(
        default=False,
        description="True if this is a large container chunk (not queried directly)",
    )
    token_count: int = Field(default=0, description="Estimated token count (~len/4)")
    content_hash: str = Field(description="SHA-256 hash of chunk content for deduplication")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Ranked retrieval candidate with fusion provenance and citation metadata.

    Retrieved candidates undergo Reciprocal Rank Fusion (RRF) and Cross-Encoder
    reranking. This model tracks intermediate scores so engineers can diagnose
    why a particular chunk was ranked first.
    """

    chunk_id: str
    document_id: str
    tenant_id: str
    collection_id: str
    content: str
    expanded_content: str | None = Field(
        default=None,
        description="Expanded parent chunk content if small-to-big retrieval is applied",
    )
    score: float = Field(description="Final fusion or rerank score used for ranking")
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
        """Return expanded content if available, else original chunk content.

        When small-to-big retrieval is active, the generator feeds `expanded_content`
        into the LLM prompt while still referencing the specific child chunk's metadata.
        """
        return self.expanded_content if self.expanded_content is not None else self.content


class RetrievalQuery(BaseModel):
    """Parameters for hybrid retrieval execution.

    Controls how the retrieval engine combines dense semantic search and sparse BM25:
    - `dense_limit` and `sparse_limit` retrieve an initial candidate pool (e.g. 40 each).
    - `rrf_k` controls rank smoothing in Reciprocal Rank Fusion (standard default = 60).
    - `expand_parent` toggles fetching large parent chunks for enriched LLM context.
    """

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
        description="Optional additional payload filters (e.g. doc_type, acl_tags)",
    )
