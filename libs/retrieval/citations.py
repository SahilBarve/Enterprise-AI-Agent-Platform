"""Citation engine and groundedness verification for Document RAG (FR-RAG-26, FR-RAG-27, FR-RAG-28).

In enterprise AI, answers must be verifiable. Users cannot trust ungrounded summaries.
This module provides:
1. Provenance Tracking: Formats retrieved context blocks into numbered sources [Source 1], [Source 2],
   associating each with chunk ID, document UUID, page number, and section path.
2. Inline Citation Parsing: Extracts bracketed citations like [1], [2], [1, 2] from LLM text and
   maps them back to their exact source metadata.
3. Hallucination Guard: Detects unmapped citations (e.g. LLM invented [7] when only 3 sources existed).
4. Sentence-Level Claim Grounding: Audits each sentence in the answer against its cited source
   excerpt; flags unsupported statements and computes a confidence score.
5. Refusal Detection: Standardizes "I don't know" behavior when documents contain insufficient evidence.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.models import SearchResult

logger = get_logger("retrieval.citations")


class Citation(BaseModel):
    """Source provenance for inline citation references (FR-RAG-26).

    Represents a specific chunk of text used as evidence for an answer.
    When a user in the UI clicks on '[1]', the UI displays these exact details.
    """

    source_number: int = Field(description="1-indexed source number matching [N] in text")
    chunk_id: str = Field(description="Originating chunk identifier in Qdrant")
    document_id: str = Field(description="Parent document UUID")
    page_number: int = Field(default=1, description="Document page number for direct jump")
    section_path: list[str] = Field(default_factory=list, description="Section heading breadcrumbs")
    excerpt: str = Field(description="Context passage excerpt displayed in citation popover")
    title: str | None = Field(default=None, description="Document title if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata")


class GroundedAnswer(BaseModel):
    """Synthesized RAG answer with inline citations and groundedness audit (FR-RAG-26, FR-RAG-27).

    Returned to API clients, supervisory agents, or web frontends.
    """

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

    # Phrases indicating the model was unable to answer due to missing evidence
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
        """Format retrieval candidates into numbered prompt blocks and build citation index.

        Example Output Prompt Block:
            [Source 1] (Document: doc-123, Page: 4, Section: Architecture > Databases)
            PgBouncer connection pooling is configured with max_client_conn=1000.

        Args:
            candidates: Ranked list of SearchResult objects.

        Returns:
            Tuple of (formatted_context_string, list_of_Citation_objects).
        """
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
        """Extract [N] citation markers from answer text and map to source citations.

        How it works:
        1. Regex searches for patterns like '[1]', '[2]', '[1, 2]', '[1][2]'.
        2. Compares cited numbers against available sources.
        3. If cited number exists in available sources -> adds to matched citations.
        4. If cited number does NOT exist (e.g. model cited [8] when only 2 sources exist) ->
           flags it as an unmapped hallucinated reference!

        Args:
            answer: Generated text containing inline brackets.
            available_citations: Citations provided in the prompt context.

        Returns:
            Tuple of (matched_citations, unmapped_numbers).
        """
        citation_map = {c.source_number: c for c in available_citations}
        # Regex matches brackets containing integers separated by commas or whitespace
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
        """Check if answer indicates insufficient evidence (FR-RAG-28).

        Args:
            answer: Generated text response.

        Returns:
            True if answer matches known refusal patterns.
        """
        lower = answer.lower()
        return any(phrase in lower for phrase in self.REFUSAL_PHRASES)

    def verify_groundedness(
        self,
        answer: str,
        citations: list[Citation],
    ) -> tuple[float, list[str]]:
        """Verify claim-level grounding of answer against cited excerpts (FR-RAG-27).

        How it works:
        1. If the answer is a refusal ("I don't know"), confidence is 1.0 (correct refusal).
        2. Splits answer into individual sentences.
        3. For each sentence, checks if it cites specific sources [N].
           - If it cites [1], checks that key informative words (length >= 4) in that sentence
             actually appear in Source 1's excerpt.
           - If coverage is < 50%, marks the sentence as an 'unsupported_claim'.
        4. Calculates confidence score = (grounded_sentences / total_sentences).

        Args:
            answer: Answer text with inline citations.
            citations: Matched citations used in the answer.

        Returns:
            Tuple of (confidence_score, unsupported_claims_list).
        """
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

            # Determine relevant reference text for this specific sentence
            if cited_nums:
                target_text = " ".join(
                    citation_map.get(num, "") for num in cited_nums if num in citation_map
                )
            else:
                target_text = all_context

            # Strip citation brackets before extracting word tokens
            s_clean = re.sub(r"\[[0-9,\s]+\]", "", s_lower)
            s_tokens = set(re.findall(r"\w{4,}", s_clean))

            if not s_tokens:
                grounded_count += 1
                continue

            # Compute term grounding ratio against target excerpt
            found_tokens = sum(1 for tok in s_tokens if tok in target_text)
            coverage = found_tokens / len(s_tokens)

            # Threshold: at least 50% of informative sentence tokens must match source text
            if coverage >= 0.5:
                grounded_count += 1
            else:
                unsupported_claims.append(sentence)

        confidence = round(grounded_count / len(sentences), 4)
        return confidence, unsupported_claims
