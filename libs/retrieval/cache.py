"""Two-tier semantic caching engine for Document RAG (FR-RAG-18 to FR-RAG-22).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Two-Tier Caching (Exact + Semantic) for Enterprise RAG?
--------------------------------------------------------------------------------
Large Language Models (LLMs) and vector retrieval pipelines are computationally
expensive and slow (often 1,500ms to 4,000ms per request).
In operational environments, many user questions are either identical or semantically
equivalent:
- Query A: "How do I increase Postgres replica connections?"
- Query B: "What is the procedure to increase PostgreSQL replica connections?"

A standard key-value cache (like Memcached or Redis exact match) treats Query A
and Query B as completely different keys. Query B results in a cache miss,
costing full latency and LLM token costs.

How Two-Tier Caching Solves This:
1. Tier 1: Exact Match (Hash-based)
   - Normalizes the query string (lowercased, whitespace-stripped).
   - Generates SHA-256 hash.
   - Performs an O(1) key lookup in Redis.
   - Latency: < 2ms.
2. Tier 2: Semantic Match (Embedding-based)
   - If Tier 1 misses, generates a dense embedding vector of the query.
   - Compares the query vector against cached query vectors for the same tenant.
   - If cosine similarity >= threshold (e.g. 0.92 or 0.95), returns the cached answer!
   - Latency: < 15ms (100x faster than calling the LLM).

Critical Enterprise Protections:
- Multi-Tenant & Scope Isolation (FR-RAG-19): Cache keys strictly incorporate
  `tenant_id`, `collection_id`, `model_name`, and `prompt_version`. Tenant A can
  never receive an answer cached by Tenant B.
- Cache Poisoning Prevention (FR-RAG-22): Answers where `is_refusal=True` (e.g.
  "I don't know") or where `confidence_score < min_confidence` are NEVER cached.
  This prevents a temporary retrieval outage or bad query from permanently poisoning
  the cache with incorrect or empty answers.
- TTL & Invalidation (FR-RAG-20, FR-RAG-21): Entries expire after a configurable TTL
  (e.g. 3600s) and support instant collection-wide or tenant-wide flushes when documents
  are updated.
================================================================================
"""

import hashlib
import time

from prometheus_client import Counter
from pydantic import BaseModel, Field

from libs.common.logging import get_logger
from libs.retrieval.citations import GroundedAnswer
from libs.retrieval.embeddings import DenseEmbeddingModel

logger = get_logger("retrieval.cache")

# Prometheus Metrics for Cache Observability (FR-RAG-20)
RAG_CACHE_HITS_TOTAL = Counter(
    "rag_cache_hits_total",
    "Total RAG cache hits partitioned by tier",
    ["tier", "tenant_id"],
)
RAG_CACHE_MISSES_TOTAL = Counter(
    "rag_cache_misses_total",
    "Total RAG cache misses",
    ["tenant_id"],
)


class CachedRAGEntry(BaseModel):
    """Payload stored in the cache representing a previously generated answer."""

    query: str
    answer: GroundedAnswer
    embedding: list[float]
    tenant_id: str
    collection_id: str
    model: str
    prompt_version: str
    created_at: float = Field(default_factory=time.time)
    cache_tier: str = "exact"
    similarity: float | None = None


class SemanticCache:
    """Two-tier exact and semantic response cache with tenant isolation and poisoning guards."""

    def __init__(
        self,
        embedding_model: DenseEmbeddingModel | None = None,
        similarity_threshold: float = 0.92,
        default_ttl_seconds: int = 3600,
        min_confidence_to_cache: float = 0.5,
        enabled: bool = True,
    ) -> None:
        """Initialize semantic cache.

        Args:
            embedding_model: Dense embedding model used to vectorize queries for semantic matching.
            similarity_threshold: Cosine similarity cutoff (e.g. 0.92) required for semantic hit.
            default_ttl_seconds: Time-to-live for cache entries in seconds.
            min_confidence_to_cache: Minimum answer confidence score required before caching.
            enabled: Master toggle to enable or disable caching per tenant/system.
        """
        self.embedding_model = embedding_model or DenseEmbeddingModel(dimension=64, use_mock=True)
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.min_confidence_to_cache = min_confidence_to_cache
        self.enabled = enabled

        # In-memory storage backing: supports testing and local execution without live Redis
        # Structure: { exact_key: (CachedRAGEntry, expiration_timestamp) }
        self._exact_store: dict[str, tuple[CachedRAGEntry, float]] = {}
        # Structure: { tenant_id: { collection_id: list[CachedRAGEntry] } }
        self._semantic_index: dict[str, dict[str, list[CachedRAGEntry]]] = {}

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query string by lowercasing and trimming excessive whitespace."""
        return " ".join(query.strip().lower().split())

    @staticmethod
    def compute_exact_hash(
        normalized_query: str,
        tenant_id: str,
        collection_id: str,
        model: str,
        prompt_version: str,
    ) -> str:
        """Generate deterministic SHA-256 key bound to query and security scope (FR-RAG-19)."""
        key_raw = f"{tenant_id}:{collection_id}:{model}:{prompt_version}:{normalized_query}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm_a = sum(a * a for a in v1) ** 0.5
        norm_b = sum(b * b for b in v2) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def get(
        self,
        query: str,
        tenant_id: str,
        collection_id: str,
        model: str = "default",
        prompt_version: str = "v1",
    ) -> CachedRAGEntry | None:
        """Lookup cached answer using Tier 1 (exact) followed by Tier 2 (semantic).

        Args:
            query: User's raw question string.
            tenant_id: Mandatory tenant isolation ID.
            collection_id: Target collection ID.
            model: Active LLM model name.
            prompt_version: Version identifier of the RAG system prompt.

        Returns:
            CachedRAGEntry if hit found and valid; None on cache miss.
        """
        if not self.enabled:
            return None

        now = time.time()
        norm_query = self.normalize_query(query)
        exact_hash = self.compute_exact_hash(
            norm_query, tenant_id, collection_id, model, prompt_version
        )

        # -------------------------------------------------------------
        # Tier 1: Exact Match Lookup (O(1))
        # -------------------------------------------------------------
        if exact_hash in self._exact_store:
            entry, expires_at = self._exact_store[exact_hash]
            if now < expires_at:
                logger.info(
                    "RAG cache hit (Tier 1: Exact)",
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    query=norm_query,
                )
                RAG_CACHE_HITS_TOTAL.labels(tier="exact", tenant_id=tenant_id).inc()
                entry.cache_tier = "exact"
                entry.similarity = 1.0
                return entry
            else:
                # Expired entry cleanup
                del self._exact_store[exact_hash]

        # -------------------------------------------------------------
        # Tier 2: Semantic Similarity Lookup (Embedding Cosine Search)
        # -------------------------------------------------------------
        candidates = (
            self._semantic_index.get(tenant_id, {}).get(collection_id, [])
        )
        if not candidates:
            RAG_CACHE_MISSES_TOTAL.labels(tenant_id=tenant_id).inc()
            return None

        # Compute embedding of the incoming query
        query_vector = self.embedding_model.embed_text(norm_query)

        best_entry: CachedRAGEntry | None = None
        best_similarity = -1.0

        for candidate in candidates:
            # Check prompt version and model compatibility
            if candidate.model != model or candidate.prompt_version != prompt_version:
                continue

            # Verify expiration
            candidate_hash = self.compute_exact_hash(
                self.normalize_query(candidate.query),
                tenant_id,
                collection_id,
                model,
                prompt_version,
            )
            if candidate_hash in self._exact_store:
                _, expires_at = self._exact_store[candidate_hash]
                if now >= expires_at:
                    continue
            else:
                continue

            sim = self.cosine_similarity(query_vector, candidate.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = candidate

        if best_entry and best_similarity >= self.similarity_threshold:
            logger.info(
                "RAG cache hit (Tier 2: Semantic)",
                tenant_id=tenant_id,
                collection_id=collection_id,
                similarity=round(best_similarity, 4),
                cached_query=best_entry.query,
                incoming_query=norm_query,
            )
            RAG_CACHE_HITS_TOTAL.labels(tier="semantic", tenant_id=tenant_id).inc()
            best_entry.cache_tier = "semantic"
            best_entry.similarity = best_similarity
            return best_entry

        # Cache Miss
        RAG_CACHE_MISSES_TOTAL.labels(tenant_id=tenant_id).inc()
        return None

    def set(
        self,
        query: str,
        answer: GroundedAnswer,
        tenant_id: str,
        collection_id: str,
        model: str = "default",
        prompt_version: str = "v1",
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store generated answer in both exact and semantic cache tiers.

        Guards (FR-RAG-22):
        - Never caches refusal responses (e.g. "I don't know based on provided docs").
        - Never caches answers below `min_confidence_to_cache`.

        Returns:
            True if cached successfully, False if blocked by poisoning guards or disabled.
        """
        if not self.enabled:
            return False

        # Guard 1: Do not cache refusals
        if answer.is_refusal:
            logger.debug("Skipping cache set: Answer is a refusal", query=query)
            return False

        # Guard 2: Do not cache low confidence answers
        if answer.confidence_score < self.min_confidence_to_cache:
            logger.debug(
                "Skipping cache set: Confidence score below threshold",
                confidence=answer.confidence_score,
                min_required=self.min_confidence_to_cache,
            )
            return False

        norm_query = self.normalize_query(query)
        exact_hash = self.compute_exact_hash(
            norm_query, tenant_id, collection_id, model, prompt_version
        )
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = time.time() + ttl

        # Compute query vector for Tier 2 semantic matching
        query_vector = self.embedding_model.embed_text(norm_query)

        entry = CachedRAGEntry(
            query=norm_query,
            answer=answer,
            embedding=query_vector,
            tenant_id=tenant_id,
            collection_id=collection_id,
            model=model,
            prompt_version=prompt_version,
            cache_tier="exact",
        )

        # 1. Store in Tier 1 exact map
        self._exact_store[exact_hash] = (entry, expires_at)

        # 2. Store in Tier 2 semantic index
        tenant_collections = self._semantic_index.setdefault(tenant_id, {})
        collection_list = tenant_collections.setdefault(collection_id, [])
        # Avoid duplicate entries for the exact same query in semantic list
        collection_list = [c for c in collection_list if c.query != norm_query]
        collection_list.append(entry)
        tenant_collections[collection_id] = collection_list

        logger.info(
            "Cached RAG answer",
            tenant_id=tenant_id,
            collection_id=collection_id,
            query=norm_query,
            ttl=ttl,
        )
        return True

    def invalidate_collection(self, tenant_id: str, collection_id: str) -> int:
        """Invalidate all cached entries for a specific collection (FR-RAG-20).

        Called automatically when documents in a collection are added, updated, or deleted.
        """
        removed_count = 0
        tenant_collections = self._semantic_index.get(tenant_id, {})
        entries = tenant_collections.pop(collection_id, [])

        for entry in entries:
            exact_hash = self.compute_exact_hash(
                entry.query, tenant_id, collection_id, entry.model, entry.prompt_version
            )
            if exact_hash in self._exact_store:
                del self._exact_store[exact_hash]
                removed_count += 1

        logger.info(
            "Invalidated collection cache",
            tenant_id=tenant_id,
            collection_id=collection_id,
            removed_entries=removed_count,
        )
        return removed_count

    def flush_tenant(self, tenant_id: str) -> int:
        """Admin flush: clear all cached responses for a tenant (FR-RAG-21)."""
        removed_count = 0
        tenant_collections = self._semantic_index.pop(tenant_id, {})
        for coll_entries in tenant_collections.values():
            for entry in coll_entries:
                exact_hash = self.compute_exact_hash(
                    entry.query, tenant_id, entry.collection_id, entry.model, entry.prompt_version
                )
                if exact_hash in self._exact_store:
                    del self._exact_store[exact_hash]
                    removed_count += 1

        logger.info(
            "Flushed tenant cache",
            tenant_id=tenant_id,
            removed_entries=removed_count,
        )
        return removed_count
