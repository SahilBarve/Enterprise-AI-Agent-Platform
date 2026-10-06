"""Information retrieval and generation evaluation metrics (FR-RAG-11, Evaluation Harness)."""

import math
import re


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top K."""
    if not relevant_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k_retrieved & relevant_set)
    return round(hits / len(relevant_set), 4)


def mrr_score(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-indexed rank of first hit."""
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Calculate Hit Rate@K: 1.0 if at least one relevant item is in top K, else 0.0."""
    if not relevant_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if len(top_k_retrieved & relevant_set) > 0 else 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def lexical_faithfulness(
    answer: str,
    reference_context: str,
) -> float:
    """Compute lexical token overlap ratio of answer statements with reference context."""
    a_tokens = set(re.findall(r"\w{4,}", answer.lower()))
    if not a_tokens:
        return 1.0
    ref_lower = reference_context.lower()
    matched = sum(1 for tok in a_tokens if tok in ref_lower)
    return round(matched / len(a_tokens), 4)
