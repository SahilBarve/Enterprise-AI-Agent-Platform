"""Unit tests for sparse BM25 vector encoding (FR-RAG-11)."""

import pytest
from qdrant_client.http import models as rest

from libs.retrieval.sparse import BM25SparseEncoder


@pytest.fixture
def encoder() -> BM25SparseEncoder:
    return BM25SparseEncoder()


@pytest.mark.unit
def test_sparse_encoder_empty_text(encoder: BM25SparseEncoder) -> None:
    """Validate empty or stopword-only text produces empty SparseVector."""
    vec = encoder.encode("")
    assert isinstance(vec, rest.SparseVector)
    assert vec.indices == []
    assert vec.values == []

    assert encoder.encode("the and is in at").indices == []


@pytest.mark.unit
def test_sparse_encoder_valid_text(encoder: BM25SparseEncoder) -> None:
    """Validate text produces sorted indices and positive BM25 weights."""
    text = "Enterprise AI Operations Platform multi-agent system"
    vec = encoder.encode(text)

    assert len(vec.indices) > 0
    assert len(vec.indices) == len(vec.values)
    # Must be sorted in ascending order for Qdrant
    assert vec.indices == sorted(vec.indices)
    assert all(v > 0 for v in vec.values)


@pytest.mark.unit
def test_sparse_encoder_batch(encoder: BM25SparseEncoder) -> None:
    """Validate batch encoding returns matching count."""
    texts = [
        "PostgreSQL checkpoint persistence",
        "LangGraph supervisor planning node",
    ]
    vectors = encoder.encode_batch(texts)
    assert len(vectors) == 2
    assert all(isinstance(v, rest.SparseVector) for v in vectors)
