"""Cross-encoder reranker and MMR diversity re-selector (FR-RAG-14, FR-RAG-15).

Why Reranking is necessary in Production RAG:
- First-stage retrieval (bi-encoders and BM25) is fast and scans millions of vectors in milliseconds,
  but it considers query and document independently.
- Cross-encoders pass both [Query, Document] through full self-attention layers together,
  allowing every word in the query to attend to every word in the candidate chunk.
- This captures nuance, negation, word order, and context that bi-encoders miss.
- By retrieving ~40 candidates and reranking down to the top 5–8, we get both high speed
  and high precision.

This module also implements Maximal Marginal Relevance (MMR) (FR-RAG-15):
- Often, adjacent chunks from the same document repeat identical sentences or boilerplate.
- MMR penalizes candidates that are too similar to already selected chunks, ensuring the top-K
  presents a diverse set of facts to the LLM.
"""

import re
from typing import Any

from libs.common.logging import get_logger
from libs.retrieval.models import SearchResult

logger = get_logger("retrieval.reranker")


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity between two texts for MMR redundancy penalty.

    Formula:
        Jaccard(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|

    Returns:
        Float in [0.0, 1.0], where 1.0 means identical word sets and 0.0 means completely disjoint.
    """
    toks1 = set(re.findall(r"\w+", text1.lower()))
    toks2 = set(re.findall(r"\w+", text2.lower()))
    if not toks1 or not toks2:
        return 0.0
    intersection = len(toks1 & toks2)
    union = len(toks1 | toks2)
    return intersection / union if union > 0 else 0.0


class CrossEncoderReranker:
    """Reranks candidate chunks with cross-encoder scoring and MMR diversity re-selection."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_mock: bool = True,
    ) -> None:
        """Initialize reranker.

        Args:
            model_name: Cross-encoder HuggingFace model name (e.g. BAAI/bge-reranker-base).
            use_mock: If True, uses deterministic fallback scorer for instant unit testing.
        """
        self.model_name = model_name
        self.use_mock = use_mock
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily load SentenceTransformers CrossEncoder model if not in mock mode."""
        if self._model is None and not self.use_mock:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(
                    "SentenceTransformers CrossEncoder unavailable; using fallback scorer",
                    error=str(e),
                )
                self.use_mock = True
        return self._model

    def _score_pair_fallback(self, query: str, content: str) -> float:
        """Deterministic, high-fidelity relevance scoring of query against candidate content.

        Evaluates 3 linguistic signals:
        1. Unigram Coverage: Fraction of query terms present in candidate chunk.
        2. Bigram Overlap: Consecutive two-word phrases matched in sequence (checks phrase order).
        3. Exact Substring Match: Bonus if full query phrase appears verbatim in the chunk.

        Combines them into a normalized score bounded in [0.0, 1.0].
        """
        q_tokens = re.findall(r"\w+", query.lower())
        c_tokens = re.findall(r"\w+", content.lower())
        if not q_tokens or not c_tokens:
            return 0.0

        q_set = set(q_tokens)
        c_set = set(c_tokens)

        # 1. Unigram coverage
        overlap = len(q_set & c_set)
        coverage = overlap / len(q_set)

        # 2. Exact phrase match bonus
        q_str = " ".join(q_tokens)
        c_str = " ".join(c_tokens)
        phrase_bonus = 0.3 if q_str in c_str else 0.0

        # 3. Bigram sequence overlap
        q_bigrams = {
            f"{q_tokens[i]}_{q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)
        }
        c_bigrams = {
            f"{c_tokens[i]}_{c_tokens[i + 1]}" for i in range(len(c_tokens) - 1)
        }
        bigram_score = (
            len(q_bigrams & c_bigrams) / len(q_bigrams) if q_bigrams else coverage
        )

        # Combined normalized score bounded between [0.0, 1.0]
        raw = 0.45 * coverage + 0.35 * bigram_score + phrase_bonus
        return min(round(raw, 4), 1.0)

    def score_candidates(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[float]:
        """Compute cross-encoder relevance scores for query-candidate pairs.

        Args:
            query: User search text.
            candidates: List of SearchResult items to score.

        Returns:
            List of float scores corresponding to each candidate.
        """
        if not candidates:
            return []

        model = self._get_model()
        if model is not None and not self.use_mock:
            pairs = [[query, c.content] for c in candidates]
            raw_scores = model.predict(pairs)
            return [float(s) for s in raw_scores]

        # Use fallback deterministic scorer
        return [self._score_pair_fallback(query, c.content) for c in candidates]

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 8,
        min_score: float | None = None,
        diversity_weight: float | None = None,
    ) -> list[SearchResult]:
        """Rerank candidates with cross-encoder scoring, score cutoff, and optional MMR (FR-RAG-14, FR-RAG-15).

        Pipeline:
        1. Scores all candidates against query.
        2. Filters out candidates below `min_score` threshold.
        3. Sorts remaining candidates in descending order.
        4. If diversity_weight is specified, applies Maximal Marginal Relevance (MMR) greedy selection:
             MMR(d) = λ * Relevance(d) - (1 - λ) * MaxSimilarity(d, already_selected)
        5. Returns top_k diverse, high-relevance items.

        Args:
            query: Query text.
            candidates: Candidate SearchResult objects.
            top_k: Target number of items to return.
            min_score: Minimum relevance score threshold (cuts off irrelevant hits).
            diversity_weight: λ parameter in [0.0, 1.0]. Lower values favor diversity; None disables MMR.

        Returns:
            Top-K reranked SearchResult objects.
        """
        if not candidates:
            return []

        # 1. Deduplicate candidates by chunk_id (preserving original candidate order)
        seen_chunks: set[str] = set()
        unique_candidates: list[SearchResult] = []
        for cand in candidates:
            if cand.chunk_id not in seen_chunks:
                seen_chunks.add(cand.chunk_id)
                unique_candidates.append(cand)

        # 2. Score all unique candidates
        scores = self.score_candidates(query, unique_candidates)
        for cand, score in zip(unique_candidates, scores, strict=True):
            cand.rerank_score = score
            cand.score = score

        # 3. Filter by min_score
        valid_candidates = (
            [c for c in unique_candidates if c.rerank_score is not None and c.rerank_score >= min_score]
            if min_score is not None
            else list(unique_candidates)
        )

        if not valid_candidates:
            return []

        # 3. Sort by score descending
        valid_candidates.sort(
            key=lambda x: x.rerank_score if x.rerank_score is not None else -1.0,
            reverse=True,
        )

        # 4. If MMR diversity is disabled, return top_k
        if diversity_weight is None or diversity_weight >= 1.0 or len(valid_candidates) <= 1:
            return valid_candidates[:top_k]

        # 5. Maximal Marginal Relevance (MMR) re-selection (FR-RAG-15)
        lambda_val = max(0.0, min(diversity_weight, 1.0))
        selected: list[SearchResult] = [valid_candidates[0]]
        remaining = valid_candidates[1:]

        while remaining and len(selected) < top_k:
            best_mmr = -float("inf")
            best_idx = -1

            for idx, cand in enumerate(remaining):
                relevance = cand.rerank_score or 0.0
                # Maximum similarity to any chunk already picked
                max_sim = max(
                    compute_token_jaccard(cand.content, s.content) for s in selected
                )
                # MMR formula: trade off relevance vs redundancy
                mmr_score = lambda_val * relevance - (1.0 - lambda_val) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected
