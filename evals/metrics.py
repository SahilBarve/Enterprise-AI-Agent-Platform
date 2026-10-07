"""Information retrieval and generation evaluation metrics (FR-RAG-11, Evaluation Harness).

This module implements formal Information Retrieval (IR) and QA evaluation metrics:
1. Recall@K: What percentage of all ground-truth relevant chunks were retrieved in top K?
2. MRR (Mean Reciprocal Rank): How close to rank 1 was the first relevant chunk?
3. Hit Rate@K: Did the retriever return at least one relevant chunk in top K (binary success)?
4. NDCG@K: Normalized Discounted Cumulative Gain (penalizes relevant items that appear low down).
5. Lexical Faithfulness: Checks if statements in the answer are grounded in reference context.
"""

import math
import re


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top K.

    Formula:
        Recall@K = |retrieved[:k] ∩ relevant| / |relevant|

    Interpretation:
        - 1.0 means ALL relevant documents were found.
        - 0.5 means only half of the necessary evidence was retrieved.
        - 0.0 means the search missed every relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs from the search engine.
        relevant_ids: Ground-truth list of chunk IDs known to contain the answer.
        k: Cutoff rank depth (default: 10).

    Returns:
        Float score in [0.0, 1.0].
    """
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
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-indexed rank of first hit.

    Formula:
        RR = 1 / rank_of_first_relevant_item

    Examples:
        - First relevant document is at Rank 1 -> RR = 1/1 = 1.0.
        - First relevant document is at Rank 2 -> RR = 1/2 = 0.5.
        - First relevant document is at Rank 4 -> RR = 1/4 = 0.25.
        - Not found in results -> RR = 0.0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth list of relevant chunk IDs.

    Returns:
        Float score in [0.0, 1.0].
    """
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
    """Calculate Hit Rate@K: 1.0 if at least one relevant item is in top K, else 0.0.

    Also known as Success@K. Useful for determining if an answer was theoretically
    possible given the retrieved context.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth list of relevant chunk IDs.
        k: Cutoff depth (default: 10).

    Returns:
        1.0 if hit, 0.0 if complete miss.
    """
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
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.

    Why NDCG matters:
    Unlike simple hit rate, NDCG heavily rewards placing the most relevant documents
    at the very top of the list. A document at rank 1 provides full value (discount = log2(2) = 1.0),
    while a document at rank 10 is discounted by log2(11) ≈ 3.46.

    Formula:
        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
        NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth list of relevant chunk IDs.
        k: Cutoff depth (default: 10).

    Returns:
        Float score in [0.0, 1.0].
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG: Best possible ranking where all relevant items appear first
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def lexical_faithfulness(
    answer: str,
    reference_context: str,
) -> float:
    """Compute lexical token overlap ratio of answer statements with reference context.

    Detects if the answer is grounded in the retrieved text or contains hallucinated vocabulary.

    Args:
        answer: Generated text response.
        reference_context: Retrieved source document text.

    Returns:
        Float ratio in [0.0, 1.0].
    """
    a_tokens = set(re.findall(r"\w{4,}", answer.lower()))
    if not a_tokens:
        return 1.0
    ref_lower = reference_context.lower()
    matched = sum(1 for tok in a_tokens if tok in ref_lower)
    return round(matched / len(a_tokens), 4)
