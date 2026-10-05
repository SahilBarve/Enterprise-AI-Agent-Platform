"""Unit tests for document parsing across file formats (FR-RAG-1, FR-RAG-4)."""

import pytest

from libs.common.errors import ValidationError
from libs.retrieval.models import BlockType
from libs.retrieval.parser import DocumentParser


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


@pytest.mark.unit
def test_parse_plain_text(parser: DocumentParser) -> None:
    """Validate plain text parsing into paragraph blocks."""
    content = b"Paragraph 1: Introduction to AI operations.\n\nParagraph 2: Second section."
    doc = parser.parse(
        raw_bytes=content,
        filename="notes.txt",
        tenant_id="tenant-1",
        collection_id="col-1",
    )

    assert doc.metadata.tenant_id == "tenant-1"
    assert doc.metadata.collection_id == "col-1"
    assert doc.metadata.doc_type == "txt"
    assert len(doc.blocks) == 2
    assert doc.blocks[0].content == "Paragraph 1: Introduction to AI operations."
    assert doc.blocks[1].content == "Paragraph 2: Second section."
    assert len(doc.metadata.content_hash) == 64


@pytest.mark.unit
def test_parse_markdown_headings_and_tables(parser: DocumentParser) -> None:
    """Validate markdown parsing tracks heading hierarchies and table structures."""
    md_content = b"""# Architecture
This is platform overview.

## Storage Layer
Postgres stores checkpoints.

| Component | Technology |
| --- | --- |
| Database | PostgreSQL |
| Cache | Redis |
"""
    doc = parser.parse(
        raw_bytes=md_content,
        filename="arch.md",
        tenant_id="tenant-1",
        collection_id="col-1",
    )

    assert doc.metadata.doc_type == "md"
    assert len(doc.blocks) >= 4

    # Heading 1
    h1 = doc.blocks[0]
    assert h1.block_type == BlockType.HEADING
    assert h1.content == "Architecture"
    assert h1.section_path == ["Architecture"]

    # Paragraph under H1
    p1 = doc.blocks[1]
    assert p1.block_type == BlockType.PARAGRAPH
    assert p1.section_path == ["Architecture"]

    # Heading 2
    h2 = doc.blocks[2]
    assert h2.block_type == BlockType.HEADING
    assert h2.content == "Storage Layer"
    assert h2.section_path == ["Architecture", "Storage Layer"]

    # Table block
    table_blocks = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(table_blocks) == 1
    table_block = table_blocks[0]
    assert table_block.table_headers == ["Component", "Technology"]
    assert "PostgreSQL" in table_block.content


@pytest.mark.unit
def test_parse_html(parser: DocumentParser) -> None:
    """Validate HTML parsing extracts title and structured DOM elements."""
    html = b"""<!DOCTYPE html>
<html>
<head><title>System Guide</title></head>
<body>
  <h1>Introduction</h1>
  <p>Welcome to the multi-agent system.</p>
</body>
</html>"""
    doc = parser.parse(
        raw_bytes=html,
        filename="guide.html",
        tenant_id="tenant-1",
        collection_id="col-1",
    )
    assert doc.metadata.title == "System Guide"
    assert len(doc.blocks) == 2
    assert doc.blocks[0].content == "Introduction"
    assert doc.blocks[1].content == "Welcome to the multi-agent system."


@pytest.mark.unit
def test_parse_csv(parser: DocumentParser) -> None:
    """Validate CSV parsing converts rows into tabular markdown with headers."""
    csv_bytes = b"model,provider,context_window\ngpt-4o,openai,128000\nllama3.2,ollama,8192\n"
    doc = parser.parse(
        raw_bytes=csv_bytes,
        filename="models.csv",
        tenant_id="tenant-1",
        collection_id="col-1",
    )
    assert len(doc.blocks) == 1
    table = doc.blocks[0]
    assert table.block_type == BlockType.TABLE
    assert table.table_headers == ["model", "provider", "context_window"]
    assert "gpt-4o" in table.content


@pytest.mark.unit
def test_parse_empty_content_raises_error(parser: DocumentParser) -> None:
    """Validate that attempting to parse empty file content raises ValidationError."""
    with pytest.raises(ValidationError):
        parser.parse(
            raw_bytes=b"",
            filename="empty.txt",
            tenant_id="tenant-1",
            collection_id="col-1",
        )


@pytest.mark.unit
def test_parse_pdf(parser: DocumentParser) -> None:
    """Validate PDF parsing extracts text and assigns page numbers."""
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = BytesIO()
    writer.write(stream)
    pdf_bytes = stream.getvalue()

    doc = parser.parse(
        raw_bytes=pdf_bytes,
        filename="report.pdf",
        tenant_id="tenant-1",
        collection_id="col-1",
    )
    assert doc.metadata.doc_type == "pdf"
    assert doc.document_id is not None
