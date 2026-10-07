"""Layout-aware multi-format document parser (FR-RAG-1, FR-RAG-4, FR-RAG-6).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why is layout-aware parsing critical before chunking?
--------------------------------------------------------------------------------
Most naive RAG implementations take raw file bytes, decode them to a single giant
string, and split arbitrarily by character count. This causes severe quality issues:
1. Mangled Tables: A table split midway across rows loses its column headers, making
   numeric rows in later chunks completely uninterpretable by an LLM.
2. Lost Hierarchy: Headings (H1, H2, H3) provide crucial context. A chunk that says
   "Set value to 0 to disable" is meaningless unless the chunk knows its parent
   section was "# Security > Authentication > Multi-Factor Bypass".
3. Citation Provenance: Without tracking PDF page numbers or document sections at
   parse time, inline citations cannot point the user to the exact page where
   the fact originated.

How this parser works:
- Computes a deterministic SHA-256 content hash on raw bytes for deduplication (FR-RAG-7).
- Dispatches parsing based on file extension:
  * Markdown: Regex matches headings (H1-H6), adjusts the section path stack, and
    buffers markdown tables into discrete `ParsedBlock` items.
  * HTML: BeautifulSoup traverses DOM elements (headings, paragraphs, tables, code pre)
    and extracts page titles.
  * CSV: Converts tabular CSV rows into standardized markdown tables with header rows.
  * PDF: Uses `pypdf` to extract text page-by-page, stamping each block with its
    1-indexed `page_number` for precise inline citations.
================================================================================
"""

import csv
import hashlib
import io
import re
import uuid
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

from libs.common.errors import ValidationError
from libs.retrieval.models import BlockType, DocumentMetadata, ParsedBlock, ParsedDocument


class DocumentParser:
    """Extracts structured layout blocks and metadata from diverse file formats."""

    @staticmethod
    def compute_hash(raw_bytes: bytes) -> str:
        """Compute SHA-256 content hash for deduplication and versioning (FR-RAG-7).

        By hashing the raw bytes prior to decoding or parsing, we can check if
        a file has already been ingested without having to parse it first.
        """
        return hashlib.sha256(raw_bytes).hexdigest()

    def parse(
        self,
        raw_bytes: bytes,
        filename: str,
        tenant_id: str,
        collection_id: str,
        acl_tags: list[str] | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """Parse raw file bytes into a structured ParsedDocument.

        Args:
            raw_bytes: Raw file content bytes.
            filename: Original filename (used to infer file extension and title).
            tenant_id: Mandatory tenant scope for isolation.
            collection_id: Target collection identifier.
            acl_tags: Optional role-based access control tags.
            custom_fields: Optional arbitrary user-defined metadata dictionary.

        Returns:
            ParsedDocument with metadata and an ordered list of ParsedBlock objects.

        Raises:
            ValidationError: If raw_bytes is empty.
        """
        if not raw_bytes:
            raise ValidationError("Cannot parse empty file content")

        # Step 1: Compute hash and basic file properties
        content_hash = self.compute_hash(raw_bytes)
        ext = Path(filename).suffix.lower().lstrip(".")
        doc_id = str(uuid.uuid4())

        metadata = DocumentMetadata(
            tenant_id=tenant_id,
            collection_id=collection_id,
            title=Path(filename).stem,
            doc_type=ext or "unknown",
            acl_tags=acl_tags or [],
            content_hash=content_hash,
            custom_fields=custom_fields or {},
        )

        # Step 2: Route file format to appropriate parser implementation
        match ext:
            case "md" | "markdown":
                blocks = self._parse_markdown(raw_bytes.decode("utf-8", errors="replace"))
            case "txt":
                blocks = self._parse_text(raw_bytes.decode("utf-8", errors="replace"))
            case "html" | "htm":
                blocks, extracted_title = self._parse_html(
                    raw_bytes.decode("utf-8", errors="replace")
                )
                if extracted_title:
                    metadata.title = extracted_title
            case "csv":
                blocks = self._parse_csv(raw_bytes.decode("utf-8", errors="replace"))
            case "pdf":
                blocks, pdf_metadata = self._parse_pdf(raw_bytes)
                if pdf_metadata.get("title"):
                    metadata.title = pdf_metadata["title"]
                if pdf_metadata.get("author"):
                    metadata.author = pdf_metadata["author"]
            case "json":
                blocks = self._parse_text(raw_bytes.decode("utf-8", errors="replace"))
            case _:
                # Fallback to UTF-8 plain text paragraph decoding
                blocks = self._parse_text(raw_bytes.decode("utf-8", errors="replace"))

        return ParsedDocument(
            document_id=doc_id,
            filename=filename,
            content_type=ext,
            metadata=metadata,
            blocks=blocks,
        )

    def _parse_text(self, text: str) -> list[ParsedBlock]:
        """Split plain text into paragraph blocks using double-newline boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        blocks: list[ParsedBlock] = []
        for p in paragraphs:
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.PARAGRAPH,
                    content=p,
                    page_number=1,
                    section_path=[],
                )
            )
        return blocks

    def _parse_markdown(self, markdown_text: str) -> list[ParsedBlock]:
        """Parse markdown tracking headings hierarchy and buffering tables.

        How Heading Hierarchy Tracking Works:
        - When an H1 `# Intro` is encountered, `current_section_path` becomes `["Intro"]`.
        - When an H2 `## Architecture` follows, it becomes `["Intro", "Architecture"]`.
        - When an H2 `## Testing` follows next, the stack pops "Architecture" and appends
          "Testing", resulting in `["Intro", "Testing"]`.
        - Every subsequent paragraph inherits this full breadcrumb path!
        """
        lines = markdown_text.splitlines()
        blocks: list[ParsedBlock] = []
        current_section_path: list[str] = []
        buffer: list[str] = []
        table_buffer: list[str] = []
        in_table = False

        def flush_paragraph() -> None:
            """Flush accumulated paragraph lines into a ParsedBlock."""
            nonlocal buffer
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.PARAGRAPH,
                            content=content,
                            page_number=1,
                            section_path=list(current_section_path),
                        )
                    )
                buffer = []

        def flush_table() -> None:
            """Flush accumulated markdown table lines into a TABLE ParsedBlock."""
            nonlocal table_buffer, in_table
            if table_buffer:
                first_line = table_buffer[0]
                # Extract column headers from the first pipe-delimited row
                headers = [col.strip() for col in first_line.strip("|").split("|")]
                content = "\n".join(table_buffer).strip()
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.TABLE,
                        content=content,
                        page_number=1,
                        section_path=list(current_section_path),
                        table_headers=headers,
                    )
                )
                table_buffer = []
            in_table = False

        for line in lines:
            stripped = line.strip()

            # 1. Heading detection (# H1, ## H2, etc.)
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_table()
                flush_paragraph()
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Pop headings deeper than or equal to current level to maintain hierarchy tree
                while len(current_section_path) >= level:
                    current_section_path.pop()
                current_section_path.append(heading_text)

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.HEADING,
                        content=heading_text,
                        page_number=1,
                        section_path=list(current_section_path),
                        heading_level=level,
                    )
                )
                continue

            # 2. Markdown table detection (| col1 | col2 |)
            if stripped.startswith("|") and stripped.endswith("|"):
                flush_paragraph()
                in_table = True
                table_buffer.append(stripped)
                continue
            elif in_table:
                # Table ended on this line
                flush_table()

            # 3. Empty line indicates paragraph break
            if not stripped:
                flush_paragraph()
            else:
                buffer.append(line)

        # Flush any remaining content buffers
        flush_table()
        flush_paragraph()
        return blocks

    def _parse_html(self, html_text: str) -> tuple[list[ParsedBlock], str | None]:
        """Parse HTML extracting structural tags and page title via BeautifulSoup."""
        soup = BeautifulSoup(html_text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        blocks: list[ParsedBlock] = []
        current_section_path: list[str] = []

        # Find significant structural elements in document order
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "pre"]):
            tag_name = element.name.lower()

            if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
                level = int(tag_name[1])
                text = element.get_text(strip=True)
                if not text:
                    continue
                while len(current_section_path) >= level:
                    current_section_path.pop()
                current_section_path.append(text)

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.HEADING,
                        content=text,
                        page_number=1,
                        section_path=list(current_section_path),
                        heading_level=level,
                    )
                )

            elif tag_name == "p":
                text = element.get_text(strip=True)
                if text:
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.PARAGRAPH,
                            content=text,
                            page_number=1,
                            section_path=list(current_section_path),
                        )
                    )

            elif tag_name == "table":
                headers = [th.get_text(strip=True) for th in element.find_all("th")]
                rows = []
                for tr in element.find_all("tr"):
                    row_cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if row_cells:
                        rows.append(" | ".join(row_cells))
                content = ""
                if headers:
                    content += " | ".join(headers) + "\n" + "--- | " * len(headers) + "\n"
                content += "\n".join(rows)

                if content.strip():
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.TABLE,
                            content=content.strip(),
                            page_number=1,
                            section_path=list(current_section_path),
                            table_headers=headers if headers else None,
                        )
                    )

            elif tag_name == "pre":
                text = element.get_text()
                if text.strip():
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.CODE,
                            content=text.strip(),
                            page_number=1,
                            section_path=list(current_section_path),
                        )
                    )

        return blocks, title

    def _parse_csv(self, csv_text: str) -> list[ParsedBlock]:
        """Parse CSV preserving tabular headers and converting rows into markdown table format.

        Why convert CSV to Markdown tables?
        - Both dense embedding models and generative LLMs have seen millions of
          markdown tables in pre-training. Markdown formatting provides clear row/column
          spatial delineation for attention heads.
        """
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            return []

        headers = [h.strip() for h in rows[0]]
        blocks: list[ParsedBlock] = []

        # Convert CSV to markdown-formatted table block
        md_table_lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for row in rows[1:]:
            if any(cell.strip() for cell in row):
                md_table_lines.append(" | ".join(cell.strip() for cell in row))

        blocks.append(
            ParsedBlock(
                block_type=BlockType.TABLE,
                content="\n".join(md_table_lines),
                page_number=1,
                section_path=["Dataset"],
                table_headers=headers,
            )
        )
        return blocks

    def _parse_pdf(self, raw_bytes: bytes) -> tuple[list[ParsedBlock], dict[str, str]]:
        """Parse PDF page-by-page preserving 1-indexed page numbers (FR-RAG-4).

        Page preservation is non-negotiable for auditability: when the system
        cites `[Source 1, Page 4]`, the user must be able to open the source PDF
        and verify the statement on page 4.
        """
        reader = PdfReader(io.BytesIO(raw_bytes))
        blocks: list[ParsedBlock] = []
        pdf_meta: dict[str, str] = {}

        if reader.metadata:
            if reader.metadata.title:
                pdf_meta["title"] = str(reader.metadata.title)
            if reader.metadata.author:
                pdf_meta["author"] = str(reader.metadata.author)

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.PARAGRAPH,
                        content=p,
                        page_number=page_idx,
                        section_path=[],
                    )
                )

        return blocks, pdf_meta
