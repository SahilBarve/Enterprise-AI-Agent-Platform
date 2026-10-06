"""Dense vector embedding generator with batching and normalization (FR-RAG-11)."""

import hashlib
import math
from typing import Any


class DenseEmbeddingModel:
    """Computes dense semantic embeddings using FastEmbed or deterministic fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 32,
        use_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.use_mock = use_mock
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily initialize FastEmbed model if not in mock mode."""
        if self._model is None and not self.use_mock:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name)
            except Exception:
                # Fall back gracefully to deterministic vectors if fastembed fails/offline
                self.use_mock = True
        return self._model

    def _mock_embed(self, text: str) -> list[float]:
        """Generate deterministic pseudo-random unit vector from text hash."""
        vec: list[float] = []
        for i in range(self.dimension):
            h = hashlib.sha256(f"{text}_{i}".encode()).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    def embed_text(self, text: str) -> list[float]:
        """Generate dense vector for a single string."""
        if self.use_mock:
            return self._mock_embed(text)

        model = self._get_model()
        if self.use_mock or model is None:
            return self._mock_embed(text)

        try:
            embeddings = list(model.embed([text], batch_size=1))
            return [float(x) for x in embeddings[0]]
        except Exception:
            return self._mock_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors for multiple strings in batches."""
        if not texts:
            return []

        if self.use_mock:
            return [self._mock_embed(t) for t in texts]

        model = self._get_model()
        if self.use_mock or model is None:
            return [self._mock_embed(t) for t in texts]

        try:
            results: list[list[float]] = []
            for emb in model.embed(texts, batch_size=self.batch_size):
                results.append([float(x) for x in emb])
            return results
        except Exception:
            return [self._mock_embed(t) for t in texts]
