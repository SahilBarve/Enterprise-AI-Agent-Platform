"""Benchmark runner evaluating dense vs. sparse vs. hybrid vs. rerank on golden datasets (Phase 1 Exit Criteria).

What is an Ablation Study in AI Engineering?
An ablation study isolates individual components of a system to prove their contribution:
1. 'dense_only': Bi-encoder semantic embeddings only. Fast, but struggles with exact codes/numbers.
2. 'sparse_only': BM25 keyword weights only. Perfect for exact terms, but blind to synonyms.
3. 'hybrid': Merges dense and sparse using Reciprocal Rank Fusion (RRF). Combines both strengths.
4. 'hybrid_rerank': Applies cross-encoder reranking on top of hybrid candidates. Maximizes MRR.

This runner indexes the golden evaluation dataset into Qdrant and measures:
- Recall@10
- Mean Reciprocal Rank (MRR)
- Hit Rate@10
- NDCG@10
- Search Latency (ms)
"""

import json
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from evals.metrics import hit_rate_at_k, mrr_score, ndcg_at_k, recall_at_k
from libs.common.logging import get_logger
from libs.retrieval.embeddings import DenseEmbeddingModel
from libs.retrieval.hybrid import HybridRetriever
from libs.retrieval.indexer import QdrantHybridIndexer
from libs.retrieval.models import Chunk, RetrievalQuery
from libs.retrieval.reranker import CrossEncoderReranker
from libs.retrieval.sparse import BM25SparseEncoder

logger = get_logger("evals.runner")


class EvaluationRunner:
    """Runs automated retrieval evaluations across multiple search configurations."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        dataset_path: Path | str | None = None,
    ) -> None:
        """Initialize evaluation runner with vector store and dataset.

        Args:
            client: QdrantClient instance (defaults to fast in-memory instance for testing).
            dataset_path: Path to golden_rag.json dataset file.
        """
        self.client = client or QdrantClient(location=":memory:")
        self.dense_model = DenseEmbeddingModel(dimension=64, use_mock=True)
        self.sparse_encoder = BM25SparseEncoder()
        self.indexer = QdrantHybridIndexer(
            client=self.client,
            dense_model=self.dense_model,
            sparse_encoder=self.sparse_encoder,
        )
        self.retriever = HybridRetriever(
            client=self.client,
            dense_model=self.dense_model,
            sparse_encoder=self.sparse_encoder,
            indexer=self.indexer,
        )
        self.reranker = CrossEncoderReranker(use_mock=True)

        # Load golden dataset from disk
        path = (
            Path(dataset_path)
            if dataset_path
            else Path(__file__).parent / "datasets" / "golden_rag.json"
        )
        with open(path, encoding="utf-8") as f:
            self.dataset: dict[str, Any] = json.load(f)

    def setup_index(self, collection_name: str = "golden_eval_col") -> int:
        """Index the golden dataset chunks into Qdrant.

        Converts raw dictionary documents into typed Chunk models and upserts them
        into Qdrant with dual vectors.

        Args:
            collection_name: Target Qdrant collection name.

        Returns:
            Number of indexed chunks.
        """
        raw_docs = self.dataset.get("documents", [])
        chunks: list[Chunk] = []

        for d in raw_docs:
            chunk = Chunk(
                id=d["id"],
                document_id=d["document_id"],
                tenant_id=d["tenant_id"],
                collection_id=d["collection_id"],
                content=d["content"],
                page_number=d.get("page_number", 1),
                section_path=d.get("section_path", []),
                content_hash=d["content_hash"],
            )
            chunks.append(chunk)

        indexed = self.indexer.index_chunks(collection_name=collection_name, chunks=chunks)
        return indexed

    def evaluate_mode(
        self,
        mode: str,
        collection_name: str = "golden_eval_col",
        top_k: int = 10,
    ) -> dict[str, float]:
        """Evaluate a specific retrieval configuration across all queries in the golden dataset.

        Ablation Modes:
        - 'dense_only': dense_weight=1.0, sparse_weight=0.0
        - 'sparse_only': dense_weight=0.0, sparse_weight=1.0
        - 'hybrid': dense_weight=1.0, sparse_weight=1.0 (RRF fusion)
        - 'hybrid_rerank': hybrid retrieval followed by Cross-Encoder reranker

        Args:
            mode: One of 'dense_only', 'sparse_only', 'hybrid', 'hybrid_rerank'.
            collection_name: Qdrant collection name.
            top_k: Number of retrieved results evaluated per query.

        Returns:
            Dict containing mean recall_at_k, mrr, hit_rate, ndcg_at_k, and avg_latency_ms.
        """
        queries = self.dataset.get("queries", [])
        recalls: list[float] = []
        mrrs: list[float] = []
        hits: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []

        for q in queries:
            query_text = q["query"]
            relevant_ids: list[str] = q["relevant_chunk_ids"]

            t0 = time.perf_counter()

            # Configure weights and search based on ablation mode
            if mode == "dense_only":
                r_query = RetrievalQuery(
                    query=query_text,
                    tenant_id="tenant-ops",
                    collection_id="eval-col",
                    top_k=top_k,
                    dense_weight=1.0,
                    sparse_weight=0.0,
                )
                results = self.retriever.search(r_query, collection_name=collection_name)
            elif mode == "sparse_only":
                r_query = RetrievalQuery(
                    query=query_text,
                    tenant_id="tenant-ops",
                    collection_id="eval-col",
                    top_k=top_k,
                    dense_weight=0.0,
                    sparse_weight=1.0,
                )
                results = self.retriever.search(r_query, collection_name=collection_name)
            elif mode == "hybrid":
                r_query = RetrievalQuery(
                    query=query_text,
                    tenant_id="tenant-ops",
                    collection_id="eval-col",
                    top_k=top_k,
                    dense_weight=1.0,
                    sparse_weight=1.0,
                )
                results = self.retriever.search(r_query, collection_name=collection_name)
            elif mode == "hybrid_rerank":
                # Retrieve top_k * 2 candidates for reranker pool
                r_query = RetrievalQuery(
                    query=query_text,
                    tenant_id="tenant-ops",
                    collection_id="eval-col",
                    top_k=top_k * 2,
                    dense_weight=1.0,
                    sparse_weight=1.0,
                )
                candidates = self.retriever.search(r_query, collection_name=collection_name)
                results = self.reranker.rerank(
                    query=query_text,
                    candidates=candidates,
                    top_k=top_k,
                )
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")

            latency_ms = (time.perf_counter() - t0) * 1000
            retrieved_ids = [r.chunk_id for r in results]

            recalls.append(recall_at_k(retrieved_ids, relevant_ids, k=top_k))
            mrrs.append(mrr_score(retrieved_ids, relevant_ids))
            hits.append(hit_rate_at_k(retrieved_ids, relevant_ids, k=top_k))
            ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k=top_k))
            latencies.append(latency_ms)

        # Aggregate across all test queries
        return {
            "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "hit_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        }

    def run_full_benchmark(
        self, collection_name: str = "golden_eval_col"
    ) -> dict[str, dict[str, float]]:
        """Run complete ablation benchmark across all retrieval modes.

        Executes all 4 modes on the indexed golden dataset and logs comparative results.

        Args:
            collection_name: Qdrant collection name to use for benchmark.

        Returns:
            Dictionary mapping each mode name to its performance metrics dict.
        """
        self.setup_index(collection_name)
        modes = ["dense_only", "sparse_only", "hybrid", "hybrid_rerank"]
        report: dict[str, dict[str, float]] = {}

        for m in modes:
            metrics = self.evaluate_mode(m, collection_name=collection_name)
            report[m] = metrics
            logger.info("Evaluated retrieval mode", mode=m, **metrics)

        return report
