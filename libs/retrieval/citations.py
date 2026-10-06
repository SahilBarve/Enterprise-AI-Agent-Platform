"""Citation engine and groundedness verification for Document RAG (FR-RAG-26, FR-RAG-27, FR-RAG-28)."""

import re
from typing import Any

from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.models import SearchResult

logger = get_logger("retrieval.citations")


class Citation(BaseModel):
    """Source provenance for inline citation references (FR-RAG-26)."""

    source_number: int = Field(description="1-indexed source number matching [N] in text")
    chunk_id: str = Field(description="Originating chunk identifier")
    document_id: str = Field(description="Parent document UUID")
    page_number: int = Field(default=1, description="Document page number")
    section_path: list[str] = Field(default_factory=list, description="Section breadcrumbs")
    excerpt: str = Field(description="Context passage excerpt")
    title: str | None = Field(default=None, description="Document title if available")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundedAnswer(BaseModel):
    """Synthesized RAG answer with inline citations and groundedness audit (FR-RAG-26, FR-RAG-27)."""

    answer: str = Field(description="Generated answer with inline [N] citations")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Matched citations corresponding to references in answer",
    )
    confidence_score: float = Field(
        default=1.0,
        description="Groundedness confidence score [0.0, 1.0]",
    )
    is_refusal: bool = Field(
        default=False,
        description="True if model answered 'I don't know' due to insufficient evidence (FR-RAG-28)",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Statements in answer lacking sufficient context grounding",
    )
    unmapped_citations: list[int] = Field(
        default_factory=list,
        description="Hallucinated citation numbers not present in provided context",
    )
    tokens_used: int = Field(default=0, description="Tokens consumed in prompt and completion")


class CitationEngine:
    """Manages context prompt formatting, citation extraction, and claim grounding verification."""

    REFUSAL_PHRASES: tuple[str, ...] = (
        "i don't know",
        "i do not know",
        "insufficient information",
        "not mentioned in the provided documents",
        "not provided in the context",
        "insufficient evidence",
    )

    def format_context(
        self, candidates: list[SearchResult]
    ) -> tuple[str, list[Citation]]:
        """Format retrieval candidates into numbered prompt blocks and build citation index."""
        context_blocks: list[str] = []
        citations: list[Citation] = []

        for idx, cand in enumerate(candidates, start=1):
            excerpt = cand.effective_content.strip()
            citation = Citation(
                source_number=idx,
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                page_number=cand.page_number,
                section_path=cand.section_path,
                excerpt=excerpt[:300] + ("..." if len(excerpt) > 300 else ""),
                title=cand.metadata.get("title"),
                metadata=cand.metadata,
            )
            citations.append(citation)

            section_str = " > ".join(cand.section_path) if cand.section_path else "General"
            block = (
                f"[Source {idx}] (Document: {cand.document_id}, Page: {cand.page_number}, Section: {section_str})\n"
                f"{excerpt}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        return formatted_context, citations

    def extract_citations(
        self, answer: str, available_citations: list[Citation]
    ) -> tuple[list[Citation], list[int]]:
        """Extract [N] citation markers from answer text and map to source citations."""
        citation_map = {c.source_number: c for c in available_citations}
        # Match patterns like [1], [2], [1, 2], [1][2]
        raw_matches = re.findall(r"\[([0-9,\s]+)\]", answer)
        cited_numbers: set[int] = set()

        for match in raw_matches:
            for num_str in match.split(","):
                num_str = num_str.strip()
                if num_str.isdigit():
                    cited_numbers.add(int(num_str))

        matched: list[Citation] = []
        unmapped: list[int] = []

        for num in sorted(cited_numbers):
            if num in citation_map:
                matched.append(citation_map[num])
            else:
                unmapped.append(num)

        return matched, unmapped

    def is_refusal(self, answer: str) -> bool:
        """Check if answer indicates insufficient evidence (FR-RAG-28)."""
        lower = answer.lower()
        return any(phrase in lower for phrase in self.REFUSAL_PHRASES)

    def verify_groundedness(
        self,
        answer: str,
        citations: list[Citation],
    ) -> tuple[float, list[str]]:
        """Verify claim-level grounding of answer against cited excerpts (FR-RAG-27)."""
        if self.is_refusal(answer):
            return 1.0, []

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", answer)
            if s.strip() and len(s.strip()) > 10
        ]
        if not sentences:
            return 1.0, []

        citation_map = {c.source_number: c.excerpt.lower() for c in citations}
        all_context = " ".join(c.excerpt.lower() for c in citations)

        unsupported_claims: list[str] = []
        grounded_count = 0

        for sentence in sentences:
            s_lower = sentence.lower()
            cited_nums = [
                int(n.strip())
                for match in re.findall(r"\[([0-9,\s]+)\]", sentence)
                for n in match.split(",")
                if n.strip().isdigit()
            ]

            # Determine relevant reference text for this sentence
            if cited_nums:
                target_text = " ".join(
                    citation_map.get(num, "") for num in cited_nums if num in citation_map
                )
            else:
                target_text = all_context

            s_clean = re.sub(r"\[[0-9,\s]+\]", "", s_lower)
            s_tokens = set(re.findall(r"\w{4,}", s_clean))

            if not s_tokens:
                grounded_count += 1
                continue

            # Compute term grounding ratio against target excerpt
            found_tokens = sum(1 for tok in s_tokens if tok in target_text)
            coverage = found_tokens / len(s_tokens)

            if coverage >= 0.5:
                grounded_count += 1
            else:
                unsupported_claims.append(sentence)

        confidence = round(grounded_count / len(sentences), 4)
        return confidence, unsupported_claims
