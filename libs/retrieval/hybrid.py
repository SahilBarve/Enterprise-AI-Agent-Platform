"""Hybrid dense + sparse vector retriever with Reciprocal Rank Fusion (FR-RAG-11, FR-RAG-13, FR-RAG-16)."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from libs.common.logging import get_logger
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantHybridIndexer
from libs.retrieval.models import RetrievalQuery, SearchResult
from libs.retrieval.sparse import BM25SparseEncoder

logger = get_logger("retrieval.hybrid")


def calculate_rrf_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> float:
    """Calculate Reciprocal Rank Fusion score from dense and sparse rankings."""
    score = 0.0
    if dense_rank is not None and dense_rank > 0:
        score += dense_weight / (k + dense_rank)
    if sparse_rank is not None and sparse_rank > 0:
        score += sparse_weight / (k + sparse_rank)
    return round(score, 6)


class HybridRetriever:
    """Executes hybrid vector queries against Qdrant and merges via Reciprocal Rank Fusion."""

    def __init__(
        self,
        client: QdrantClient,
        dense_model: DenseEmbeddingModel | None = None,
        sparse_encoder: BM25SparseEncoder | None = None,
        indexer: QdrantHybridIndexer | None = None,
    ) -> None:
        self.client = client
        self.dense_model = dense_model or DenseEmbeddingModel(use_mock=True)
        self.sparse_encoder = sparse_encoder or BM25SparseEncoder()
        self.indexer = indexer or QdrantHybridIndexer(
            client=client,
            dense_model=self.dense_model,
            sparse_encoder=self.sparse_encoder,
        )

    def _build_filter(
        self,
        tenant_id: str,
        collection_id: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> rest.Filter:
        """Construct mandatory tenant isolation filter and optional metadata conditions."""
        must_conditions: list[rest.Condition] = [
            rest.FieldCondition(
                key="tenant_id",
                match=rest.MatchValue(value=tenant_id),
            )
        ]
        if collection_id is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="collection_id",
                    match=rest.MatchValue(value=collection_id),
                )
            )

        # Do not retrieve parent container chunks directly during standard search
        must_conditions.append(
            rest.FieldCondition(
                key="is_parent",
                match=rest.MatchValue(value=False),
            )
        )

        if filter_metadata:
            for key, val in filter_metadata.items():
                if isinstance(val, list):
                    must_conditions.append(
                        rest.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest.MatchAny(any=val),
                        )
                    )
                else:
                    must_conditions.append(
                        rest.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest.MatchValue(value=val),
                        )
                    )

        return rest.Filter(must=must_conditions)

    def search(
        self,
        query: RetrievalQuery,
        collection_name: str | None = None,
    ) -> list[SearchResult]:
        """Perform dual dense + sparse search and fuse results with Reciprocal Rank Fusion."""
        col_name = collection_name or self.indexer.get_collection_name(
            query.tenant_id, query.collection_id
        )

        # Verify collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        if col_name not in collections:
            logger.warning("Target collection does not exist", collection=col_name)
            return []

        query_filter = self._build_filter(
            tenant_id=query.tenant_id,
            collection_id=query.collection_id,
            filter_metadata=query.filter_metadata,
        )

        # 1. Dense search
        dense_vec = self.dense_model.embed_text(query.query)
        dense_results = self.client.query_points(
            collection_name=col_name,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=query.dense_limit,
            with_payload=True,
        )

        # 2. Sparse search
        sparse_vec = self.sparse_encoder.encode(query.query)
        sparse_results = self.client.query_points(
            collection_name=col_name,
            query=rest.SparseVector(
                indices=sparse_vec.indices,
                values=sparse_vec.values,
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=query.sparse_limit,
            with_payload=True,
        )

        # 3. Track ranks and payloads
        dense_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for rank, point in enumerate(dense_results.points, start=1):
            pid = str(point.id)
            dense_ranks[pid] = rank
            dense_scores[pid] = float(point.score)
            if point.payload:
                payloads[pid] = dict(point.payload)

        sparse_ranks: dict[str, int] = {}
        sparse_scores: dict[str, float] = {}
        for rank, point in enumerate(sparse_results.points, start=1):
            pid = str(point.id)
            sparse_ranks[pid] = rank
            sparse_scores[pid] = float(point.score)
            if point.payload and pid not in payloads:
                payloads[pid] = dict(point.payload)

        # 4. Fuse scores using Reciprocal Rank Fusion (RRF)
        candidate_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        scored_candidates: list[SearchResult] = []

        for cid in candidate_ids:
            payload = payloads.get(cid, {})
            rrf_score = calculate_rrf_score(
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid),
                k=query.rrf_k,
                dense_weight=query.dense_weight,
                sparse_weight=query.sparse_weight,
            )

            result = SearchResult(
                chunk_id=payload.get("chunk_id", cid),
                document_id=payload.get("document_id", ""),
                tenant_id=payload.get("tenant_id", query.tenant_id),
                collection_id=payload.get("collection_id", query.collection_id),
                content=payload.get("content", ""),
                score=rrf_score,
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid),
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
                page_number=payload.get("page_number", 1),
                section_path=payload.get("section_path", []),
                parent_id=payload.get("parent_id"),
                metadata=payload.get("metadata", {}),
            )
            scored_candidates.append(result)

        # Sort descending by RRF fusion score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        top_candidates = scored_candidates[: query.top_k]

        # 5. Parent Context Expansion (FR-RAG-16)
        if query.expand_parent:
            self._expand_parent_context(col_name, top_candidates)

        logger.info(
            "Hybrid retrieval completed",
            collection=col_name,
            total_candidates=len(candidate_ids),
            returned=len(top_candidates),
        )
        return top_candidates

    def _expand_parent_context(
        self,
        collection_name: str,
        results: list[SearchResult],
    ) -> None:
        """Batch-retrieve parent chunk contents and populate expanded_content."""
        parent_ids = list({r.parent_id for r in results if r.parent_id})
        if not parent_ids:
            return

        try:
            parent_points = self.client.retrieve(
                collection_name=collection_name,
                ids=parent_ids,
                with_payload=True,
            )
            parent_map: dict[str, str] = {}
            for p in parent_points:
                if p.payload and "content" in p.payload:
                    parent_map[str(p.id)] = str(p.payload["content"])

            for res in results:
                if res.parent_id and res.parent_id in parent_map:
                    res.expanded_content = parent_map[res.parent_id]
        except Exception as e:
            logger.warning("Failed to expand parent chunks", error=str(e))
