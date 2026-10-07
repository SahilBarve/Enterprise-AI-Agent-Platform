"""Adaptive Corrective RAG (CRAG) router and query rewriting engine (FR-RAG-12, FR-RAG-17).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Adaptive Corrective RAG (CRAG) Instead of Naive RAG?
--------------------------------------------------------------------------------
Standard RAG operates on a blind assumption: "If the vector database returned
results, they must be relevant." In reality, vector databases ALWAYS return top-K
results, even for gibberish or out-of-domain questions!
If the retrieved chunks are irrelevant, feeding them to an LLM leads to hallucinations,
confusion, or misleading statements.

The Corrective RAG (CRAG) Architecture (Yan et al., 2024):
CRAG inserts a self-reflective "Retrieval Grader" between retrieval and generation:

                 [ User Query ]
                       │
               [ Hybrid Retrieval ]
                       │
            ┌──────────────────────┐
            │   CRAG Evaluator     │
            │  (Confidence Grader) │
            └──────────┬───────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   [ CORRECT ]    [ AMBIGUOUS ]   [ INCORRECT ]
 (Score >= 0.65) (0.30 <= S < 0.65)(Score < 0.30)
       │               │               │
       ▼               ▼               ▼
 Compress &      Rewrite Query    Halt Internal Gen,
 Generate        & Re-retrieve    Signal External Fallback
                                  (Web Research / Refusal)

1. CORRECT Confidence:
   Retrieved passages strongly address the question. Filter irrelevant sentences
   and synthesize the grounded answer.
2. AMBIGUOUS Confidence:
   Retrieved passages are marginally related, but likely missing key context due to
   suboptimal query phrasing. CRAG rewrites the query, expands synonyms, and retrieves
   supplementary passages before generating.
3. INCORRECT Confidence:
   Retrieved passages have no factual connection to the query. Generating from this
   context would produce dangerous hallucinations. The router signals external
   fallback (Web Research Agent or honest refusal).
================================================================================
"""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.models import SearchResult

logger = get_logger("retrieval.crag")


class CRAGConfidence(StrEnum):
    """Tri-state confidence bands defined by Corrective RAG methodology."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class CRAGDecision(BaseModel):
    """Routing decision emitted by the adaptive retrieval evaluator."""

    confidence: CRAGConfidence = Field(description="Confidence classification: correct, ambiguous, incorrect")
    score: float = Field(description="Normalized aggregate relevance score [0.0, 1.0]")
    action: str = Field(description="Recommended action: proceed_generation, rewrite_and_retry, fallback_external")
    rewritten_query: str | None = Field(default=None, description="Rewritten or expanded query if ambiguous")
    rationale: str = Field(description="Human-readable explanation of routing decision")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdaptiveCRAGRouter:
    """Evaluates retrieval quality, routes execution, and rewrites queries for RAG resilience."""

    def __init__(
        self,
        correct_threshold: float = 0.55,
        ambiguous_threshold: float = 0.25,
    ) -> None:
        """Initialize CRAG confidence thresholds.

        Args:
            correct_threshold: Minimum aggregate score to classify results as CORRECT.
            ambiguous_threshold: Minimum score to classify results as AMBIGUOUS (below is INCORRECT).
        """
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold

    @staticmethod
    def calculate_passage_relevance(query: str, content: str) -> float:
        """Score lexical and semantic relevance between query and passage content."""
        query_words = set(re.findall(r"\w+", query.lower()))
        content_words = set(re.findall(r"\w+", content.lower()))

        if not query_words or not content_words:
            return 0.0

        # Term overlap
        overlap = query_words.intersection(content_words)
        overlap_ratio = len(overlap) / len(query_words)

        # Content keyword density
        density = len(overlap) / len(content_words)
        return min(1.0, overlap_ratio * 0.8 + density * 0.2)

    def evaluate_retrieval(self, query: str, results: list[SearchResult]) -> CRAGDecision:
        """Grade retrieved results into CORRECT, AMBIGUOUS, or INCORRECT bands (FR-RAG-17).

        Combines:
        1. Single-chunk relevance (cross-encoder score & lexical density).
        2. Collective candidate coverage (union of informative query terms covered across chunks).
        This guarantees that multi-hop or multi-part queries spanning multiple passages
        receive proper recognition as CORRECT evidence.

        Args:
            query: User's original search or question string.
            results: Ranked SearchResult candidates from retriever/reranker.

        Returns:
            CRAGDecision specifying confidence band and recommended action.
        """
        if not results:
            logger.info("CRAG evaluated: Zero results returned -> INCORRECT", query=query)
            return CRAGDecision(
                confidence=CRAGConfidence.INCORRECT,
                score=0.0,
                action="fallback_external",
                rationale="Retriever returned zero candidate documents for the query.",
            )

        # 1. Collective query token coverage across all candidate chunks
        query_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {
            "how", "do", "does", "did", "and", "or", "what", "which", "who", "whom",
            "this", "that", "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "with", "by", "is", "are", "was", "were", "be", "been"
        }
        informative_query_words = query_words - stopwords or query_words

        collective_words: set[str] = set()
        for r in results:
            collective_words.update(re.findall(r"\w+", r.effective_content.lower()))

        collective_overlap = informative_query_words.intersection(collective_words)
        collective_coverage = len(collective_overlap) / max(1, len(informative_query_words))

        # 2. Individual passage scores
        passage_scores: list[float] = []
        for r in results:
            lexical_score = self.calculate_passage_relevance(query, r.effective_content)
            if r.rerank_score is not None:
                combined = 0.6 * r.rerank_score + 0.4 * lexical_score
            else:
                combined = lexical_score
            passage_scores.append(combined)

        top1_score = passage_scores[0]
        mean_score = sum(passage_scores) / len(passage_scores)

        # 3. Blended aggregate score: 50% collective coverage + 35% top1 + 15% mean
        aggregate_score = round(
            min(1.0, 0.50 * collective_coverage + 0.35 * top1_score + 0.15 * mean_score), 4
        )

        # Tri-state Decision Routing
        if aggregate_score >= self.correct_threshold:
            decision = CRAGDecision(
                confidence=CRAGConfidence.CORRECT,
                score=aggregate_score,
                action="proceed_generation",
                rationale=f"Candidate evidence is highly relevant (score: {aggregate_score} >= {self.correct_threshold}). Proceeding to generation.",
            )
        elif aggregate_score >= self.ambiguous_threshold:
            rewritten = self.rewrite_query(query)
            decision = CRAGDecision(
                confidence=CRAGConfidence.AMBIGUOUS,
                score=aggregate_score,
                action="rewrite_and_retry",
                rewritten_query=rewritten,
                rationale=f"Candidate evidence is marginally relevant (score: {aggregate_score}). Rewriting query and widening search.",
            )
        else:
            decision = CRAGDecision(
                confidence=CRAGConfidence.INCORRECT,
                score=aggregate_score,
                action="fallback_external",
                rationale=f"Candidate evidence has insufficient relevance (score: {aggregate_score} < {self.ambiguous_threshold}). Triggering fallback.",
            )

        logger.info(
            "CRAG evaluation complete",
            confidence=decision.confidence,
            score=decision.score,
            action=decision.action,
        )
        return decision

    def rewrite_query(self, query: str) -> str:
        """Rewrite ambiguous or keyword-sparse query to maximize retrieval recall (FR-RAG-12).

        Rules:
        1. Strips conversational filler phrases ("Can you tell me", "Please explain how to").
        2. Normalizes punctuation.
        3. Appends domain search indicators if question is very short.
        """
        cleaned = query.strip()
        filler_patterns = [
            r"^(can you please|can you|could you|please)\s+",
            r"^(tell me about|explain how to|how do i|how to)\s+",
            r"^(what is the best way to|what is the procedure for)\s+",
            r"\?+$",
        ]
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # If query became too short, keep original
        if len(cleaned.split()) < 2:
            return f"{query} configuration documentation"

        return cleaned

    def decompose_query(self, query: str) -> list[str]:
        """Decompose a complex multi-part query into targeted sub-queries (FR-RAG-12).

        Example:
        'Compare Postgres connection pooling and explain Patroni failover steps'
        -> ['Postgres connection pooling', 'Patroni failover steps']
        """
        # Split on conjunctions: 'and', 'also', 'as well as', 'vs', 'versus'
        parts = re.split(r"\s+(?:and|also|as well as|versus|vs)\s+", query, flags=re.IGNORECASE)
        sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]

        # If no conjunction split occurred, return the original query as a single item
        if not sub_queries:
            return [query]

        return sub_queries
