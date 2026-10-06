"""Unit and regression tests for evaluation harness and baseline benchmark (Phase 1 Exit Criteria)."""

import pytest

from evals.metrics import (
    hit_rate_at_k,
    lexical_faithfulness,
    mrr_score,
    ndcg_at_k,
    recall_at_k,
)
from evals.runner import EvaluationRunner


@pytest.mark.unit
def test_recall_at_k() -> None:
    """Validate Recall@K calculation."""
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = ["c2", "c4"]
    # At k=2: retrieved=['c1', 'c2'] -> 1 hit out of 2 = 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    # At k=4: 2 hits out of 2 = 1.0
    assert recall_at_k(retrieved, relevant, k=4) == 1.0
    # No relevant items in top 1
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


@pytest.mark.unit
def test_mrr_score() -> None:
    """Validate Mean Reciprocal Rank (MRR) score calculation."""
    # First relevant item at rank 1 -> 1.0
    assert mrr_score(["c1", "c2"], ["c1"]) == 1.0
    # First relevant item at rank 2 -> 0.5
    assert mrr_score(["c1", "c2"], ["c2"]) == 0.5
    # First relevant item at rank 4 -> 0.25
    assert mrr_score(["c1", "c2", "c3", "c4"], ["c4"]) == 0.25
    # No match -> 0.0
    assert mrr_score(["c1", "c2"], ["c9"]) == 0.0


@pytest.mark.unit
def test_hit_rate_and_ndcg() -> None:
    """Validate Hit Rate and NDCG calculations."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c2"]

    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0

    ndcg_val = ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 < ndcg_val <= 1.0


@pytest.mark.unit
def test_lexical_faithfulness() -> None:
    """Validate lexical faithfulness overlap ratio."""
    context = "PostgreSQL utilizes PgBouncer for high throughput connection pooling."
    answer = "PgBouncer is used for PostgreSQL connection pooling."
    score = lexical_faithfulness(answer, context)
    assert score >= 0.8


@pytest.mark.unit
def test_golden_dataset_baseline_benchmark() -> None:
    """Run full benchmark across ablation modes to establish Recall@10 baseline (Phase 1 Exit Criteria)."""
    runner = EvaluationRunner()
    report = runner.run_full_benchmark(collection_name="eval_baseline_test")

    # Verify report structure
    assert "dense_only" in report
    assert "sparse_only" in report
    assert "hybrid" in report
    assert "hybrid_rerank" in report

    # Baseline verification
    hybrid_rerank_metrics = report["hybrid_rerank"]
    assert hybrid_rerank_metrics["recall_at_k"] >= 0.80
    assert hybrid_rerank_metrics["hit_rate"] >= 0.80
    assert hybrid_rerank_metrics["mrr"] >= 0.50
    assert hybrid_rerank_metrics["avg_latency_ms"] < 200.0
