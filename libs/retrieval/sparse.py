"""Sparse BM25 vector encoder for Qdrant hybrid retrieval (FR-RAG-11).

This module implements BM25 (Best Matching 25) sparse lexical vector encoding.
While dense vectors capture general conceptual meaning, BM25 captures exact keyword
matches (e.g. error codes, product numbers, names). By computing both and indexing
them in Qdrant, the platform achieves best-in-class retrieval precision.
"""

import hashlib
import math
import re
from collections import Counter

from qdrant_client.http import models as rest


class BM25SparseEncoder:
    """Computes sparse BM25 vectors for Qdrant sparse vector indexing.

    Converts raw text into token indices and BM25 term weights, matching Qdrant's
    SparseVector(indices=..., values=...) schema for inverted index scoring.
    """

    # Common English stopwords ignored during tokenization to prevent uninformative noise
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
        "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "whom", "why", "with",
        "would", "you", "your",
    }

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_len: float = 100.0,
        max_vocab_size: int = 1_000_000,
    ) -> None:
        """Initialize the BM25 encoder with standard Information Retrieval parameters.

        Args:
            k1: Term frequency saturation parameter (typically 1.2 to 2.0).
                Controls how quickly additional occurrences of a word give diminishing returns.
            b: Document length normalization parameter (typically 0.75).
                Penalizes very long documents so short, focused chunks aren't drowned out.
            avg_doc_len: Expected average chunk length in tokens (used for length normalization).
            max_vocab_size: Maximum index space for the hashing trick (modulo 1,000,000).
        """
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        self.max_vocab_size = max_vocab_size

    def tokenize(self, text: str) -> list[str]:
        """Normalize and tokenize text into alphanumeric word tokens.

        Steps:
        1. Convert text to lowercase.
        2. Extract alphanumeric sequences with hyphens and underscores (at least 2 chars).
        3. Filter out stopwords to retain only content-bearing terms.

        Args:
            text: Raw input string.

        Returns:
            List of cleaned word tokens.
        """
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def _token_to_index(self, token: str) -> int:
        """Deterministic hashing of token into a positive 31-bit index.

        Uses the 'Hashing Trick' (Feature Hashing):
        Instead of keeping a huge dictionary of words synchronized across multiple
        servers, we hash each word to a unique integer index in [0, max_vocab_size).
        Every server computes the exact same index with zero coordination.

        Args:
            token: Normalized word token.

        Returns:
            Integer index in range [0, max_vocab_size).
        """
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.max_vocab_size

    def encode(self, text: str) -> rest.SparseVector:
        """Encode text into a Qdrant SparseVector with BM25 term weights.

        How it works:
        1. Tokenizes text and counts frequency of each word (tf).
        2. Computes the BM25 saturation formula for each term:
           weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        3. Applies logarithmic smoothing: log(1 + weight).
        4. Maps each term to its hashed index and accumulates weights if collisions occur.
        5. Sorts indices in ascending order (mandatory requirement for Qdrant sparse vectors).

        Args:
            text: Input text string to be indexed or searched.

        Returns:
            Qdrant SparseVector object containing sorted indices and corresponding weights.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return rest.SparseVector(indices=[], values=[])

        doc_len = len(tokens)
        counts = Counter(tokens)

        # Document length normalization factor
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
        index_weights: dict[int, float] = {}

        for token, tf in counts.items():
            idx = self._token_to_index(token)
            # BM25 term frequency saturation
            weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
            # Logarithmic compression to prevent outlier domination
            score = round(math.log1p(weight), 4)

            # Accumulate weight in case of hash collisions
            index_weights[idx] = index_weights.get(idx, 0.0) + score

        # Qdrant requires sorted indices
        sorted_indices = sorted(index_weights.keys())
        sorted_values = [index_weights[i] for i in sorted_indices]

        return rest.SparseVector(
            indices=sorted_indices,
            values=sorted_values,
        )

    def encode_batch(self, texts: list[str]) -> list[rest.SparseVector]:
        """Encode multiple texts in batch for high ingestion throughput.

        Args:
            texts: List of strings to encode.

        Returns:
            List of Qdrant SparseVector instances.
        """
        return [self.encode(t) for t in texts]
