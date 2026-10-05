"""Multi-strategy document chunking engine (FR-RAG-5, FR-RAG-7, FR-RAG-16)."""

import hashlib
import uuid

from libs.retrieval.models import BlockType, Chunk, ChunkStrategy, ParsedBlock, ParsedDocument


class DocumentChunker:
    """Splits ParsedDocuments into indexable Chunks using configurable strategies."""

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size

    @staticmethod
    def compute_chunk_hash(text: str) -> str:
        """Compute SHA-256 hash for chunk deduplication (FR-RAG-7)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token count estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Dispatch document chunking based on configured strategy."""
        match self.strategy:
            case ChunkStrategy.PARENT_CHILD:
                return self._chunk_parent_child(doc)
            case ChunkStrategy.TABLE_AWARE:
                return self._chunk_table_aware(doc)
            case ChunkStrategy.HEADING_AWARE:
                return self._chunk_heading_aware(doc)
            case ChunkStrategy.RECURSIVE | _:
                return self._chunk_recursive(doc)

    def _split_text_recursively(self, text: str, max_size: int, overlap: int) -> list[str]:
        """Split text using recursive separators: paragraph, newline, sentence, word."""
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators, max_size, overlap)

    def _recursive_split(
        self, text: str, separators: list[str], max_size: int, overlap: int
    ) -> list[str]:
        if not separators or len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + (len(sep) if current else 0)

            if part_len > max_size and remaining_seps:
                # Sub-split large segment with next granular separator
                if current:
                    chunks.append(sep.join(current).strip())
                    current = []
                    current_len = 0
                sub_chunks = self._recursive_split(part, remaining_seps, max_size, overlap)
                chunks.extend(sub_chunks)
                continue

            if current_len + part_len <= max_size:
                current.append(part)
                current_len += part_len
            else:
                if current:
                    chunks.append(sep.join(current).strip())

                # Sliding window overlap
                if overlap > 0 and current:
                    overlap_buffer: list[str] = []
                    acc = 0
                    for item in reversed(current):
                        acc += len(item) + len(sep)
                        overlap_buffer.insert(0, item)
                        if acc >= overlap:
                            break
                    current = overlap_buffer + [part]
                    current_len = sum(len(x) for x in current) + len(sep) * (len(current) - 1)
                else:
                    current = [part]
                    current_len = len(part)

        if current:
            final_text = sep.join(current).strip()
            if final_text:
                chunks.append(final_text)

        return [c for c in chunks if c.strip()]

    def _chunk_recursive(self, doc: ParsedDocument) -> list[Chunk]:
        """Recursive chunking adhering to document block layout."""
        chunks: list[Chunk] = []
        chunk_idx = 0

        for block in doc.blocks:
            sub_texts = self._split_text_recursively(
                block.content, self.chunk_size, self.chunk_overlap
            )
            for sub_text in sub_texts:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        tenant_id=doc.metadata.tenant_id,
                        collection_id=doc.metadata.collection_id,
                        content=sub_text,
                        chunk_index=chunk_idx,
                        page_number=block.page_number,
                        section_path=block.section_path,
                        token_count=self.estimate_tokens(sub_text),
                        content_hash=self.compute_chunk_hash(sub_text),
                        metadata={"block_type": block.block_type},
                    )
                )
                chunk_idx += 1

        return chunks

    def _chunk_heading_aware(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunking grouped strictly by section headings."""
        chunks: list[Chunk] = []
        chunk_idx = 0

        # Group blocks by section path
        sections: dict[str, list[ParsedBlock]] = {}
        for block in doc.blocks:
            key = " > ".join(block.section_path) if block.section_path else "Root"
            sections.setdefault(key, []).append(block)

        for sec_key, blocks in sections.items():
            combined_text = "\n\n".join(b.content for b in blocks)
            sub_texts = self._split_text_recursively(
                combined_text, self.chunk_size, self.chunk_overlap
            )
            for sub_text in sub_texts:
                first_block = blocks[0]
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        tenant_id=doc.metadata.tenant_id,
                        collection_id=doc.metadata.collection_id,
                        content=sub_text,
                        chunk_index=chunk_idx,
                        page_number=first_block.page_number,
                        section_path=first_block.section_path,
                        token_count=self.estimate_tokens(sub_text),
                        content_hash=self.compute_chunk_hash(sub_text),
                        metadata={"section": sec_key},
                    )
                )
                chunk_idx += 1

        return chunks

    def _chunk_table_aware(self, doc: ParsedDocument) -> list[Chunk]:
        """Table-aware chunking preserving header rows across split table slices."""
        chunks: list[Chunk] = []
        chunk_idx = 0

        for block in doc.blocks:
            if block.block_type == BlockType.TABLE and block.table_headers:
                header_line = " | ".join(block.table_headers)
                sep_line = " | ".join(["---"] * len(block.table_headers))
                table_header = f"{header_line}\n{sep_line}"

                lines = block.content.splitlines()
                # Skip original header lines if present in content
                data_lines = [
                    line
                    for line in lines
                    if not all(c in "-| " for c in line) and line != header_line
                ]

                # Group rows preserving header in each slice
                current_rows: list[str] = []
                current_len = len(table_header)

                for row in data_lines:
                    if current_len + len(row) + 1 > self.chunk_size and current_rows:
                        slice_content = f"{table_header}\n" + "\n".join(current_rows)
                        chunks.append(
                            Chunk(
                                id=str(uuid.uuid4()),
                                document_id=doc.document_id,
                                tenant_id=doc.metadata.tenant_id,
                                collection_id=doc.metadata.collection_id,
                                content=slice_content,
                                chunk_index=chunk_idx,
                                page_number=block.page_number,
                                section_path=block.section_path,
                                token_count=self.estimate_tokens(slice_content),
                                content_hash=self.compute_chunk_hash(slice_content),
                                metadata={"is_table": True, "headers": block.table_headers},
                            )
                        )
                        chunk_idx += 1
                        current_rows = [row]
                        current_len = len(table_header) + len(row) + 1
                    else:
                        current_rows.append(row)
                        current_len += len(row) + 1

                if current_rows:
                    slice_content = f"{table_header}\n" + "\n".join(current_rows)
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            tenant_id=doc.metadata.tenant_id,
                            collection_id=doc.metadata.collection_id,
                            content=slice_content,
                            chunk_index=chunk_idx,
                            page_number=block.page_number,
                            section_path=block.section_path,
                            token_count=self.estimate_tokens(slice_content),
                            content_hash=self.compute_chunk_hash(slice_content),
                            metadata={"is_table": True, "headers": block.table_headers},
                        )
                    )
                    chunk_idx += 1
            else:
                # Regular block: recursive fallback
                sub_texts = self._split_text_recursively(
                    block.content, self.chunk_size, self.chunk_overlap
                )
                for sub_text in sub_texts:
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            tenant_id=doc.metadata.tenant_id,
                            collection_id=doc.metadata.collection_id,
                            content=sub_text,
                            chunk_index=chunk_idx,
                            page_number=block.page_number,
                            section_path=block.section_path,
                            token_count=self.estimate_tokens(sub_text),
                            content_hash=self.compute_chunk_hash(sub_text),
                            metadata={"block_type": block.block_type},
                        )
                    )
                    chunk_idx += 1

        return chunks

    def _chunk_parent_child(self, doc: ParsedDocument) -> list[Chunk]:
        """Parent-child small-to-big chunking (FR-RAG-5, FR-RAG-16).

        Produces:
        1. Large Parent chunks (parent_chunk_size) stored with is_parent=True.
        2. Small Child chunks (chunk_size) pointing to parent_id for retrieval expansion.
        """
        all_chunks: list[Chunk] = []
        chunk_idx = 0

        # Step 1: Create large parent chunks across document
        parent_texts = self._split_text_recursively(
            doc.full_text, self.parent_chunk_size, overlap=100
        )

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())
            parent_chunk = Chunk(
                id=parent_id,
                document_id=doc.document_id,
                tenant_id=doc.metadata.tenant_id,
                collection_id=doc.metadata.collection_id,
                content=p_text,
                chunk_index=chunk_idx,
                page_number=1,
                section_path=[],
                parent_id=None,
                is_parent=True,
                token_count=self.estimate_tokens(p_text),
                content_hash=self.compute_chunk_hash(p_text),
                metadata={"parent_index": p_idx},
            )
            all_chunks.append(parent_chunk)
            chunk_idx += 1

            # Step 2: Split parent chunk into smaller child chunks
            child_texts = self._split_text_recursively(p_text, self.chunk_size, self.chunk_overlap)
            for c_text in child_texts:
                child_chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    tenant_id=doc.metadata.tenant_id,
                    collection_id=doc.metadata.collection_id,
                    content=c_text,
                    chunk_index=chunk_idx,
                    page_number=1,
                    section_path=[],
                    parent_id=parent_id,
                    is_parent=False,
                    token_count=self.estimate_tokens(c_text),
                    content_hash=self.compute_chunk_hash(c_text),
                    metadata={"parent_id": parent_id},
                )
                all_chunks.append(child_chunk)
                chunk_idx += 1

        return all_chunks
