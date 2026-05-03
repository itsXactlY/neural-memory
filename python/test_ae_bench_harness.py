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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "benchmarks" / "ae_domain_memory_bench"))

from run_ae_domain_bench import (  # noqa: E402
    _category_regression_gate,
    _collect_provenance,
    _PROVENANCE_ENV_KEYS,
    run_scored,
)


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

    def test_run_scored_emits_per_query_rows(self) -> None:
        """Per-query rows are required for attributing future bench flips
        (the 0.6061 -> 0.5758 incident could not be attributed without
        per-query data; one Spanish query flipped silently)."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = [
            {"id": 100}, {"id": 200}, {"id": 300},
        ]
        queries = [{
            "id": "TEST-PQ-001", "category": "cat",
            "query": "x", "ground_truth_ids": [200],
            "expected_channels": [], "minimum_rank": 5,
            "temporal_mode": "current_or_unspecified",
        }]
        result = run_scored(mem, queries, k=3)
        self.assertIn("per_query", result)
        self.assertEqual(len(result["per_query"]), 1)
        pq = result["per_query"][0]
        self.assertEqual(pq["id"], "TEST-PQ-001")
        self.assertEqual(pq["category"], "cat")
        self.assertEqual(pq["retrieved_ids"], [100, 200, 300])
        self.assertEqual(pq["ground_truth_ids"], [200])
        self.assertEqual(pq["first_hit_rank"], 2)
        self.assertEqual(pq["hit_at_5"], 1)
        self.assertEqual(pq["hit_at_10"], 1)
        self.assertIn("latency_ms", pq)
        self.assertIsInstance(pq["latency_ms"], (int, float))

    def test_per_query_first_hit_rank_none_when_miss(self) -> None:
        """Misses must have first_hit_rank=None so attribution stays honest."""
        mem = MagicMock()
        mem.hybrid_recall.return_value = [
            {"id": 1}, {"id": 2}, {"id": 3},
        ]
        queries = [{
            "id": "TEST-MISS", "category": "c",
            "query": "x", "ground_truth_ids": [999],
            "expected_channels": [], "minimum_rank": 5,
            "temporal_mode": "current_or_unspecified",
        }]
        result = run_scored(mem, queries, k=5)
        self.assertIsNone(result["per_query"][0]["first_hit_rank"])
        self.assertEqual(result["per_query"][0]["hit_at_5"], 0)
        self.assertEqual(result["per_query"][0]["mrr"], 0.0)

    def test_collect_provenance_captures_env_flags(self) -> None:
        """The 7 env vars that affect retrieval ranking must be captured.
        Without this, env-driven flips (e.g. NM_SPANISH_TRANSLATE) cannot
        be distinguished from substrate or model drift."""
        mem = MagicMock()
        mem._db_path = "/tmp/nonexistent-test.db"
        args = MagicMock(
            db="/tmp/nonexistent-test.db", mode="scored", category=None,
            k=10, embedding_backend="auto", rerank=True,
            mmr_lambda=0.0, percentile_floor=0.0,
        )
        with patch.dict(os.environ,
                        {"NM_SPANISH_TRANSLATE": "1",
                         "NM_RERANK_ES_DISABLE": "0"},
                        clear=False):
            prov = _collect_provenance(mem, args, [])
        self.assertIn("env", prov)
        for key in _PROVENANCE_ENV_KEYS:
            self.assertIn(key, prov["env"])
        self.assertEqual(prov["env"]["NM_SPANISH_TRANSLATE"], "1")
        self.assertEqual(prov["env"]["NM_RERANK_ES_DISABLE"], "0")
        self.assertIn("git_head", prov)
        self.assertIn("ts_iso", prov)
        self.assertIn("ts_epoch", prov)
        self.assertIn("query_file_md5", prov)
        self.assertIn("substrate_counts", prov)
        self.assertIn("models", prov)
        self.assertIn("args", prov)
        self.assertEqual(prov["args"]["rerank"], True)
        self.assertEqual(prov["args"]["mode"], "scored")

    def test_category_regression_gate_flags_drop(self) -> None:
        """A >0.05 R@5 drop in any category must be flagged.
        Models the actual incident: spanish_whatsapp 0.50 -> 0.25."""
        prev = {
            "per_category": {
                "spanish_whatsapp": {"r@5": 0.50, "n": 4},
                "lennar_lots": {"r@5": 0.6667, "n": 9},
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(prev, f)
            prev_path = f.name
        try:
            current = {
                "per_category": {
                    "spanish_whatsapp": {"r@5": 0.25, "n": 4},
                    "lennar_lots": {"r@5": 0.6667, "n": 9},
                }
            }
            gate = _category_regression_gate(current, prev_path,
                                             threshold=0.05)
            self.assertTrue(gate["enabled"])
            self.assertEqual(len(gate["regressions"]), 1)
            reg = gate["regressions"][0]
            self.assertEqual(reg["category"], "spanish_whatsapp")
            self.assertEqual(reg["delta"], -0.25)
        finally:
            os.unlink(prev_path)

    def test_category_regression_gate_disabled_when_no_prev(self) -> None:
        """No --prev-results means gate is disabled, never errors."""
        gate = _category_regression_gate({"per_category": {}}, None)
        self.assertFalse(gate["enabled"])
        self.assertEqual(gate["regressions"], [])


if __name__ == "__main__":
    unittest.main()
