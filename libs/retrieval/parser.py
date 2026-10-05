"""Layout-aware multi-format document parser (FR-RAG-1, FR-RAG-4, FR-RAG-6)."""

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
        """Compute SHA-256 content hash for deduplication and versioning (FR-RAG-7)."""
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
        """Parse raw file bytes into a structured ParsedDocument."""
        if not raw_bytes:
            raise ValidationError("Cannot parse empty file content")

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
                # Fallback to UTF-8 text decoding
                blocks = self._parse_text(raw_bytes.decode("utf-8", errors="replace"))

        return ParsedDocument(
            document_id=doc_id,
            filename=filename,
            content_type=ext,
            metadata=metadata,
            blocks=blocks,
        )

    def _parse_text(self, text: str) -> list[ParsedBlock]:
        """Split plain text into paragraph blocks."""
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
        """Parse markdown tracking headings hierarchy and tables."""
        lines = markdown_text.splitlines()
        blocks: list[ParsedBlock] = []
        current_section_path: list[str] = []
        buffer: list[str] = []
        table_buffer: list[str] = []
        in_table = False

        def flush_paragraph() -> None:
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
            nonlocal table_buffer, in_table
            if table_buffer:
                headers: list[str] = []
                first_line = table_buffer[0]
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

            # Heading detection
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_table()
                flush_paragraph()
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Adjust section hierarchy
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

            # Markdown table detection
            if stripped.startswith("|") and stripped.endswith("|"):
                flush_paragraph()
                in_table = True
                table_buffer.append(stripped)
                continue
            elif in_table:
                flush_table()

            # Empty line indicates paragraph break
            if not stripped:
                flush_paragraph()
            else:
                buffer.append(line)

        flush_table()
        flush_paragraph()
        return blocks

    def _parse_html(self, html_text: str) -> tuple[list[ParsedBlock], str | None]:
        """Parse HTML extracting structural tags and page title."""
        soup = BeautifulSoup(html_text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        blocks: list[ParsedBlock] = []
        current_section_path: list[str] = []

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
        """Parse CSV preserving tabular headers and structured rows."""
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
        """Parse PDF page-by-page preserving page numbers (FR-RAG-4)."""
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
