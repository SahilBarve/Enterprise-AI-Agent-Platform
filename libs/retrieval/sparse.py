"""Sparse BM25 vector encoder for Qdrant hybrid retrieval (FR-RAG-11)."""

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

    STOPWORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
    }

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_len: float = 100.0,
        max_vocab_size: int = 1_000_000,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        self.max_vocab_size = max_vocab_size

    def tokenize(self, text: str) -> list[str]:
        """Normalize and tokenize text into alphanumeric word tokens."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def _token_to_index(self, token: str) -> int:
        """Deterministic hashing of token into a positive 31-bit index."""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.max_vocab_size

    def encode(self, text: str) -> rest.SparseVector:
        """Encode text into a Qdrant SparseVector with BM25 term weights."""
        tokens = self.tokenize(text)
        if not tokens:
            return rest.SparseVector(indices=[], values=[])

        doc_len = len(tokens)
        counts = Counter(tokens)

        # BM25 term saturation formulation: tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
        index_weights: dict[int, float] = {}

        for token, tf in counts.items():
            idx = self._token_to_index(token)
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
        """Encode multiple texts in batch."""
        return [self.encode(t) for t in texts]
