"""Unit tests for dense vector embeddings and normalization (FR-RAG-11)."""

import math

import pytest

from libs.retrieval.embeddings import DenseEmbeddingModel


@pytest.fixture
def embedding_model() -> DenseEmbeddingModel:
    return DenseEmbeddingModel(dimension=384, use_mock=True)


@pytest.mark.unit
def test_dense_embedding_dimension(embedding_model: DenseEmbeddingModel) -> None:
    """Validate embedding vector length matches target dimension."""
    text = "LangGraph supervisor agent"
    vec = embedding_model.embed_text(text)
    assert len(vec) == 384


@pytest.mark.unit
def test_dense_embedding_normalized(embedding_model: DenseEmbeddingModel) -> None:
    """Validate embedding vector is L2-normalized to unit length."""
    text = "Hybrid search Reciprocal Rank Fusion"
    vec = embedding_model.embed_text(text)
    l2_norm = math.sqrt(sum(v * v for v in vec))
    assert pytest.approx(l2_norm, abs=1e-3) == 1.0


@pytest.mark.unit
def test_dense_embedding_deterministic(embedding_model: DenseEmbeddingModel) -> None:
    """Validate identical texts produce identical vectors."""
    text = "Deterministic vector test"
    vec1 = embedding_model.embed_text(text)
    vec2 = embedding_model.embed_text(text)
    assert vec1 == vec2


@pytest.mark.unit
def test_dense_embedding_batch(embedding_model: DenseEmbeddingModel) -> None:
    """Validate batch embedding returns matching count."""
    texts = ["Document RAG", "SQL Analytics", "Web Research"]
    vectors = embedding_model.embed_batch(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
