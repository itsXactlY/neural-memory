"""Regression coverage for benchmarks/ae_domain_memory_bench/run_ae_domain_bench.py.

Caught 2026-05-01: harness was calling mem.recall() (dense-only) instead
of mem.hybrid_recall() (multi-channel). Phase 7.5 wirings (entity/
procedural/locus/stale/contradiction) couldn't influence the bench
because the dense path doesn't read them. This test locks down the
correct path.

Three contracts:
  1. run_scored() actually invokes hybrid_recall (not recall) — proves
     the path-fix from commit 9fc0d4b is durable
  2. run_scored() respects the rerank kwarg
  3. End-to-end: a labeled memory in top-K produces hit_at_5=1
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "benchmarks" / "ae_domain_memory_bench"))

from run_ae_domain_bench import run_scored  # noqa: E402


class BenchHarnessTests(unittest.TestCase):
    def test_run_scored_calls_hybrid_recall_not_recall(self) -> None:
        """Lock down commit 9fc0d4b: harness must use hybrid_recall."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = [{"id": 100}, {"id": 200}]
        mem.recall.return_value = [{"id": 999}]  # should NOT be called
        queries = [{
            "id": "TEST-001",
            "category": "test",
            "query": "what is foo",
            "ground_truth_ids": [100],
            "expected_channels": [],
            "minimum_rank": 5,
            "temporal_mode": "current_or_unspecified",
        }]
        result = run_scored(mem, queries, k=5)
        mem.hybrid_recall.assert_called_once()
        mem.recall.assert_not_called()
        self.assertEqual(result["mode"], "scored")
        self.assertEqual(result["queries_evaluated"], 1)

    def test_run_scored_passes_rerank_kwarg(self) -> None:
        """rerank=True must pass through to hybrid_recall."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = []
        queries = [{
            "id": "TEST-002",
            "category": "test",
            "query": "x",
            "ground_truth_ids": [1],
            "expected_channels": [],
            "minimum_rank": 5,
            "temporal_mode": "current_or_unspecified",
        }]
        run_scored(mem, queries, k=10, rerank=True)
        _, kwargs = mem.hybrid_recall.call_args
        self.assertEqual(kwargs.get("rerank"), True)

    def test_labeled_memory_in_top_k_produces_hit(self) -> None:
        """E2E: labeled id in retrieved → hit_at_5 = 1, mrr = 1/rank."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = [
            {"id": 7}, {"id": 42}, {"id": 100}, {"id": 200}, {"id": 300},
        ]
        queries = [{
            "id": "TEST-003",
            "category": "test",
            "query": "x",
            "ground_truth_ids": [42],
            "expected_channels": [],
            "minimum_rank": 5,
            "temporal_mode": "current_or_unspecified",
        }]
        result = run_scored(mem, queries, k=5)
        # 1 query, 1 hit, ranked at position 2 → mrr=0.5
        category = list(result["per_category"].values())[0]
        self.assertEqual(category["r@5"], 1.0)
        self.assertEqual(category["mrr"], 0.5)

    def test_unlabeled_query_skipped(self) -> None:
        """Queries with empty ground_truth_ids must be skipped + counted."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = []
        queries = [
            {"id": "L1", "category": "c", "query": "x",
             "ground_truth_ids": [1], "expected_channels": [],
             "minimum_rank": 5, "temporal_mode": "current_or_unspecified"},
            {"id": "U1", "category": "c", "query": "y",
             "ground_truth_ids": [], "expected_channels": [],
             "minimum_rank": 5, "temporal_mode": "current_or_unspecified"},
        ]
        result = run_scored(mem, queries)
        self.assertEqual(result["queries_evaluated"], 1)
        self.assertEqual(result["queries_skipped_no_ground_truth"], 1)


if __name__ == "__main__":
    unittest.main()
