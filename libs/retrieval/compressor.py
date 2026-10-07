"""Context compression, token budget manager, and 'lost in the middle' reordering (FR-RAG-23 to FR-RAG-25).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Context Compression and "Lost in the Middle" Mitigation?
--------------------------------------------------------------------------------
1. The Problem with Raw Retrieved Chunks:
   When a hybrid retriever returns 5–10 chunks, they often contain extraneous filler:
   copyright notices, standard boilerplate, or paragraphs discussing adjacent topics.
   Feeding uncompressed chunks into the LLM causes:
   - Slower generation speed (time-to-first-token scales with prompt length).
   - Higher operational cost (APIs bill per input token).
   - Context pollution (noisy irrelevant sentences increase hallucination risk).

2. Sentence-Level Relevance Filtering (FR-RAG-23):
   Instead of keeping or discarding whole chunks, the compressor splits chunks into
   individual sentences and scores each sentence against the query. Sentences that
   contain zero informative keywords or relevance to the query are pruned, keeping
   only factual, high-information sentences.

3. "Lost in the Middle" Phenomenon (FR-RAG-25):
   A landmark study by Liu et al. (Stanford/Berkeley) revealed that Transformer
   attention heads suffer from severe primacy and recency bias:
   - LLMs easily retrieve and recall facts placed at the BEGINNING of a prompt (primacy).
   - LLMs easily recall facts placed at the END of a prompt (recency).
   - Facts placed in the MIDDLE are frequently overlooked or "lost" (accuracy drops 30%+).

   How we solve it (U-shaped Reordering):
   Suppose our top-ranked candidate chunks are [Rank 1, Rank 2, Rank 3, Rank 4, Rank 5].
   Instead of sequential ordering, we arrange them as:
     Index 0: Rank 1 (Top-1 at the very start)
     Index 1: Rank 3
     Index 2: Rank 5 (Least relevant in the middle)
     Index 3: Rank 4
     Index 4: Rank 2 (Top-2 at the very end)
   Both top-ranked evidence pieces are placed in prime attention zones!

4. Token Budget Manager (FR-RAG-24):
   Strictly enforces token limits (e.g. 1500 tokens for context) to fit model windows,
   preventing context overflow and tracking compression efficiency metrics.
================================================================================
"""

import math
import re

from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.models import SearchResult

logger = get_logger("retrieval.compressor")


class CompressedContext(BaseModel):
    """Result of context compression, token budgeting, and reordering."""

    results: list[SearchResult] = Field(description="Compressed and reordered search results")
    initial_tokens: int = Field(description="Token count before compression")
    compressed_tokens: int = Field(description="Token count after compression and pruning")
    compression_ratio: float = Field(
        description="Ratio of compressed / initial tokens (lower = more compressed)"
    )
    dropped_sentences_count: int = Field(default=0, description="Number of filler sentences pruned")


class ContextCompressor:
    """Compresses retrieved context passages, enforces token budgets, and mitigates 'lost in the middle'."""

    def __init__(
        self,
        min_sentence_relevance: float = 0.05,
        default_max_tokens: int = 1500,
        enable_lost_in_middle_reorder: bool = True,
    ) -> None:
        """Initialize compressor configuration.

        Args:
            min_sentence_relevance: Minimum score threshold for a sentence to be retained.
            default_max_tokens: Maximum token budget allowed for total prompt context.
            enable_lost_in_middle_reorder: If True, applies U-shaped reordering to candidates.
        """
        self.min_sentence_relevance = min_sentence_relevance
        self.default_max_tokens = default_max_tokens
        self.enable_lost_in_middle_reorder = enable_lost_in_middle_reorder

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (~4 characters per token for English)."""
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation boundaries."""
        # Splits on period, question mark, exclamation, or newline followed by whitespace
        raw_parts = re.split(r"(?<=[.?!])\s+|\n\n+", text.strip())
        return [p.strip() for p in raw_parts if p.strip()]

    def calculate_sentence_relevance(self, query: str, sentence: str) -> float:
        """Score sentence relevance against query using token overlap and keyword density.

        Computes token overlap ratio with penalty for short generic sentences.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        sentence_words = set(re.findall(r"\w+", sentence.lower()))

        if not query_words or not sentence_words:
            return 0.0

        intersection = query_words.intersection(sentence_words)
        overlap_ratio = len(intersection) / len(query_words)

        # Give slight boost for sentences containing multiple matched query tokens
        density = len(intersection) / len(sentence_words)
        return round(overlap_ratio * 0.7 + density * 0.3, 4)

    def filter_sentences(self, query: str, content: str) -> tuple[str, int]:
        """Prune irrelevant sentences from a chunk content string (FR-RAG-23).

        Args:
            query: User's question string.
            content: Raw chunk content.

        Returns:
            Tuple of (compressed_content_string, count_of_dropped_sentences).
        """
        sentences = self.split_sentences(content)
        # If chunk is very short (1-2 sentences), keep as is to avoid over-pruning
        if len(sentences) <= 2:
            return content, 0

        kept_sentences: list[str] = []
        dropped_count = 0

        for sentence in sentences:
            score = self.calculate_sentence_relevance(query, sentence)
            if score >= self.min_sentence_relevance:
                kept_sentences.append(sentence)
            else:
                dropped_count += 1

        # If filtering accidentally dropped everything, preserve original content
        if not kept_sentences:
            return content, 0

        compressed_text = " ".join(kept_sentences)
        return compressed_text, dropped_count

    def reorder_lost_in_middle(self, results: list[SearchResult]) -> list[SearchResult]:
        """Mitigate 'Lost in the Middle' attention degradation via U-shaped reordering (FR-RAG-25).

        Places highest-relevance items at the boundaries (start and end), while
        lower-relevance items sit in the interior.

        Example:
        Given ordered list: [A (rank 1), B (rank 2), C (rank 3), D (rank 4), E (rank 5)]
        Reordered to:       [A, C, E, D, B]
        - A (Rank 1) is at the start (Index 0).
        - B (Rank 2) is at the end (Index 4).
        - C (Rank 3) is at Index 1.
        """
        if len(results) <= 2:
            return list(results)

        reordered: list[SearchResult] = []
        left: list[SearchResult] = []
        right: list[SearchResult] = []

        # Alternate distributing results to left and right boundaries
        for idx, item in enumerate(results):
            if idx % 2 == 0:
                left.append(item)
            else:
                right.append(item)

        # Reverse the right side so Rank 2 ends up at the very end
        reordered = left + list(reversed(right))
        return reordered

    def compress(
        self,
        query: str,
        results: list[SearchResult],
        max_tokens: int | None = None,
    ) -> CompressedContext:
        """Execute full compression pipeline: sentence pruning, token budgeting, and reordering.

        Args:
            query: User query string.
            results: Ranked SearchResult list from retriever/reranker.
            max_tokens: Maximum allowed token budget (defaults to self.default_max_tokens).

        Returns:
            CompressedContext with token accounting and compressed candidate list.
        """
        budget = max_tokens or self.default_max_tokens
        if not results:
            return CompressedContext(
                results=[],
                initial_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                dropped_sentences_count=0,
            )

        # Step 1: Calculate initial token count
        initial_tokens = sum(
            self.estimate_tokens(r.effective_content) for r in results
        )

        # Step 2: Sentence-level filtering per chunk (FR-RAG-23)
        compressed_results: list[SearchResult] = []
        total_dropped = 0
        current_tokens = 0

        for r in results:
            content_to_filter = r.effective_content
            pruned_text, dropped = self.filter_sentences(query, content_to_filter)
            total_dropped += dropped

            chunk_tokens = self.estimate_tokens(pruned_text)

            # Step 3: Enforce token budget (FR-RAG-24)
            if current_tokens + chunk_tokens > budget:
                # If budget is exhausted, stop including further chunks
                logger.info(
                    "Token budget reached during compression",
                    current_tokens=current_tokens,
                    budget=budget,
                )
                break

            # Create updated SearchResult with compressed content
            updated_result = r.model_copy(
                update={
                    "content": pruned_text,
                    "expanded_content": pruned_text if r.expanded_content else None,
                }
            )
            compressed_results.append(updated_result)
            current_tokens += chunk_tokens

        # Step 4: Reorder to mitigate "Lost in the Middle" (FR-RAG-25)
        if self.enable_lost_in_middle_reorder and len(compressed_results) > 2:
            final_results = self.reorder_lost_in_middle(compressed_results)
        else:
            final_results = compressed_results

        # Step 5: Final metrics accounting
        final_tokens = sum(self.estimate_tokens(r.effective_content) for r in final_results)
        ratio = round(final_tokens / max(1, initial_tokens), 4)

        logger.info(
            "Context compression complete",
            initial_tokens=initial_tokens,
            compressed_tokens=final_tokens,
            compression_ratio=ratio,
            dropped_sentences=total_dropped,
            retained_chunks=len(final_results),
        )

        return CompressedContext(
            results=final_results,
            initial_tokens=initial_tokens,
            compressed_tokens=final_tokens,
            compression_ratio=ratio,
            dropped_sentences_count=total_dropped,
        )
