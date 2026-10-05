"""Unit tests for document chunking strategies (FR-RAG-5, FR-RAG-7, FR-RAG-16)."""

import pytest

from libs.retrieval.chunker import DocumentChunker
from libs.retrieval.models import (
    BlockType,
    ChunkStrategy,
    DocumentMetadata,
    ParsedBlock,
    ParsedDocument,
)


@pytest.fixture
def sample_document() -> ParsedDocument:
    metadata = DocumentMetadata(
        tenant_id="tenant-123",
        collection_id="col-abc",
        title="Agent Architecture",
        doc_type="md",
        content_hash="mock-hash-123",
    )
    blocks = [
        ParsedBlock(
            block_type=BlockType.HEADING,
            content="Agent Overview",
            page_number=1,
            section_path=["Agent Overview"],
            heading_level=1,
        ),
        ParsedBlock(
            block_type=BlockType.PARAGRAPH,
            content=(
                "The supervisor orchestrates five specialized agents. "
                "Document RAG performs hybrid retrieval. Web research conducts deep search. "
                "SQL Analytics queries PostgreSQL. Data Processing executes sandboxed Python. "
                "Report Generation formats executive PDF briefs."
            ),
            page_number=1,
            section_path=["Agent Overview"],
        ),
    ]
    return ParsedDocument(
        document_id="doc-999",
        filename="overview.md",
        content_type="md",
        metadata=metadata,
        blocks=blocks,
    )


@pytest.mark.unit
def test_recursive_chunking(sample_document: ParsedDocument) -> None:
    """Validate recursive chunking enforces chunk size and generates metadata."""
    chunker = DocumentChunker(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == "doc-999"
        assert chunk.tenant_id == "tenant-123"
        assert chunk.collection_id == "col-abc"
        assert len(chunk.content) <= 120  # bounded by chunk size + word bounds
        assert len(chunk.content_hash) == 64
        assert chunk.token_count > 0


@pytest.mark.unit
def test_heading_aware_chunking(sample_document: ParsedDocument) -> None:
    """Validate heading-aware chunking preserves section metadata."""
    chunker = DocumentChunker(
        strategy=ChunkStrategy.HEADING_AWARE,
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert first_chunk.section_path == ["Agent Overview"]
    assert "section" in first_chunk.metadata


@pytest.mark.unit
def test_table_aware_chunking() -> None:
    """Validate table-aware chunking repeats header rows in each chunk."""
    headers = ["ID", "Metric", "Value"]
    rows = [f"{i} | Latency | {i * 10}ms" for i in range(1, 20)]
    table_content = "ID | Metric | Value\n--- | --- | ---\n" + "\n".join(rows)

    doc = ParsedDocument(
        document_id="doc-table",
        filename="metrics.md",
        content_type="md",
        metadata=DocumentMetadata(
            tenant_id="tenant-1",
            collection_id="col-1",
            content_hash="hash-table",
        ),
        blocks=[
            ParsedBlock(
                block_type=BlockType.TABLE,
                content=table_content,
                table_headers=headers,
                page_number=1,
            )
        ],
    )

    chunker = DocumentChunker(
        strategy=ChunkStrategy.TABLE_AWARE,
        chunk_size=150,
    )
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert "ID | Metric | Value" in chunk.content
        assert chunk.metadata["is_table"] is True
        assert chunk.metadata["headers"] == headers


@pytest.mark.unit
def test_parent_child_chunking(sample_document: ParsedDocument) -> None:
    """Validate parent-child small-to-big chunking links child chunks to parent_id."""
    chunker = DocumentChunker(
        strategy=ChunkStrategy.PARENT_CHILD,
        chunk_size=80,
        chunk_overlap=10,
        parent_chunk_size=300,
    )
    chunks = chunker.chunk_document(sample_document)

    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]

    assert len(parents) >= 1
    assert len(children) >= 1

    parent_ids = {p.id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.is_parent is False
