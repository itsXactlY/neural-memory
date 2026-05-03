"""Tests for S1c subset reporting + S1d comparable-label regression gate.

Run with:
    python3.11 -m unittest benchmarks.ae_domain_memory_bench.test_bench_subsets_gate -v
or from the bench dir:
    python3.11 -m unittest discover -s benchmarks/ae_domain_memory_bench -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# Import the module under test.  Works whether run from repo root or bench dir.
import sys, os
_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from run_ae_domain_bench import (
    PRESERVED_33_QUERY_IDS,
    SUBSET_38_QUERY_IDS,
    _build_subset,
    _build_subsets,
    _category_regression_gate,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_pq_row(qid: str, category: str, h5: int = 1) -> dict:
    return {
        "id": qid,
        "category": category,
        "ground_truth_ids": [1],
        "retrieved_ids": [1],
        "hit_at_5": h5,
        "hit_at_10": h5,
        "mrr": float(h5),
        "first_hit_rank": 1 if h5 else None,
        "latency_ms": 1.0,
    }


def _make_scored_result(pq_rows: list[dict]) -> dict:
    """Build a minimal scored result dict from per_query rows."""
    return {"mode": "scored", "per_query": pq_rows}


def _make_provenance(**kwargs) -> dict:
    base = {
        "query_file_md5": "abc123",
        "git_head": "deadbeef",
        "db_path": "/fake/memory.db",
        "substrate_counts": {"memories": 100, "connections_active": 50},
    }
    base.update(kwargs)
    return base


def _write_artifact(d: dict) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(d, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# S1c — Subset reporting tests
# ---------------------------------------------------------------------------

class TestSubsetBlock(unittest.TestCase):

    def _scored_with_33(self) -> dict:
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in PRESERVED_33_QUERY_IDS[:5]]
        rows += [_make_pq_row(qid, "materials_sku") for qid in PRESERVED_33_QUERY_IDS[5:10]]
        rows += [_make_pq_row(qid, "lennar_lots") for qid in PRESERVED_33_QUERY_IDS[10:]]
        return _make_scored_result(rows)

    def test_subset_label_count_preserved_33(self):
        result = self._scored_with_33()
        prov = _make_provenance()
        block = _build_subset("preserved_33", PRESERVED_33_QUERY_IDS, result, prov)
        self.assertEqual(block["label_count"], len(PRESERVED_33_QUERY_IDS),
                         "label_count must equal len(PRESERVED_33_QUERY_IDS) when all are present")

    def test_subset_label_count_38(self):
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in SUBSET_38_QUERY_IDS]
        result = _make_scored_result(rows)
        prov = _make_provenance()
        block = _build_subset("subset_38", SUBSET_38_QUERY_IDS, result, prov)
        self.assertEqual(block["label_count"], len(SUBSET_38_QUERY_IDS))

    def test_dropped_ids_populated_when_id_missing(self):
        # Use only a subset of PRESERVED_33_QUERY_IDS — the rest are "missing".
        present = list(PRESERVED_33_QUERY_IDS[:5])
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in present]
        result = _make_scored_result(rows)
        prov = _make_provenance()
        block = _build_subset("preserved_33", PRESERVED_33_QUERY_IDS, result, prov)
        expected_dropped = sorted(set(PRESERVED_33_QUERY_IDS) - set(present))
        self.assertEqual(sorted(block["dropped_ids"]), expected_dropped)

    def test_subsets_block_keys(self):
        # With all 33+extra present, subsets should have preserved_33, subset_38,
        # new_label_only, full_57.
        all_ids = list(SUBSET_38_QUERY_IDS) + ["ELC-NEW-X", "LOT-NEW-Y"]
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in all_ids]
        result = _make_scored_result(rows)
        prov = _make_provenance()
        subsets = _build_subsets(result, prov)
        self.assertIn("preserved_33", subsets)
        self.assertIn("subset_38", subsets)
        self.assertIn("new_label_only", subsets)
        self.assertIn("full_57", subsets)

    def test_subsets_full_57_label_count_matches_per_query(self):
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in SUBSET_38_QUERY_IDS]
        result = _make_scored_result(rows)
        prov = _make_provenance()
        subsets = _build_subsets(result, prov)
        self.assertEqual(subsets["full_57"]["label_count"], len(SUBSET_38_QUERY_IDS))

    def test_subset_provenance_fields_propagated(self):
        rows = [_make_pq_row(qid, "electrical_contracting") for qid in PRESERVED_33_QUERY_IDS]
        result = _make_scored_result(rows)
        prov = _make_provenance(query_file_md5="md5abc", git_head="sha1234")
        block = _build_subset("preserved_33", PRESERVED_33_QUERY_IDS, result, prov)
        self.assertEqual(block["query_md5"], "md5abc")
        self.assertEqual(block["git_head"], "sha1234")


# ---------------------------------------------------------------------------
# S1d — Comparable-label regression gate tests
# ---------------------------------------------------------------------------

class TestComparableLabelGate(unittest.TestCase):

    def _make_artifact(self, rows: list[dict]) -> str:
        d = {
            "mode": "scored",
            "per_query": rows,
            "per_category": {},
            "category_regression_gate": {"enabled": False, "regressions": [], "regression_detected": False},
        }
        return _write_artifact(d)

    def test_gate_disabled_without_prev(self):
        current = _make_scored_result([_make_pq_row("ELC-001", "electrical_contracting")])
        gate = _category_regression_gate(current, None)
        self.assertFalse(gate["enabled"])
        self.assertFalse(gate["regression_detected"])

    def test_gate_no_regression_on_comparable_intersect(self):
        # Both current and prev have same IDs, same scores → no regression.
        ids = ["ELC-001", "ELC-005", "ELC-011"]
        prev_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in ids]
        cur_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in ids]
        prev_path = self._make_artifact(prev_rows)
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertTrue(gate["enabled"])
        self.assertFalse(gate["regression_detected"])
        self.assertEqual(gate["regressions"], [])
        self.assertEqual(gate["comparable_query_count"], 3)

    def test_gate_fires_on_intersect_drop(self):
        # prev: 3 hits out of 3 = 1.0; current: 0 hits = 0.0 → drop > 0.05.
        ids = ["ELC-001", "ELC-005", "ELC-011"]
        prev_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in ids]
        cur_rows  = [_make_pq_row(qid, "electrical_contracting", h5=0) for qid in ids]
        prev_path = self._make_artifact(prev_rows)
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertTrue(gate["regression_detected"])
        self.assertEqual(len(gate["regressions"]), 1)
        self.assertEqual(gate["regressions"][0]["category"], "electrical_contracting")

    def test_gate_does_not_fire_on_label_expansion_only(self):
        # prev: 2 labeled queries in cat; current: same 2 + 3 new ones (label expansion).
        # The per-intersect scores are identical → no regression.
        shared_ids = ["ELC-001", "ELC-005"]
        new_ids    = ["ELC-009", "ELC-040", "ELC-022"]
        prev_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in shared_ids]
        cur_rows  = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in shared_ids + new_ids]
        prev_path = self._make_artifact(prev_rows)
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertFalse(gate["regression_detected"],
                         "Label expansion must not trigger regression gate")
        self.assertIn("electrical_contracting", gate["label_expansion_categories"])

    def test_gate_label_expansion_categories_reported(self):
        shared = ["ELC-001"]
        added  = ["ELC-002", "ELC-003"]
        prev_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in shared]
        cur_rows  = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in shared + added]
        prev_path = self._make_artifact(prev_rows)
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertIn("electrical_contracting", gate["label_expansion_categories"])

    def test_gate_comparable_query_count_correct(self):
        ids_a = ["ELC-001", "ELC-005"]
        ids_b = ["ELC-001", "ELC-009"]  # ELC-005 only in prev; ELC-009 only in current
        prev_rows = [_make_pq_row(qid, "electrical_contracting") for qid in ids_a]
        cur_rows  = [_make_pq_row(qid, "electrical_contracting") for qid in ids_b]
        prev_path = self._make_artifact(prev_rows)
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertEqual(gate["comparable_query_count"], 1,
                         "Only ELC-001 is in both; intersect should be 1")

    def test_gate_error_on_unreadable_prev(self):
        current = _make_scored_result([_make_pq_row("ELC-001", "electrical_contracting")])
        gate = _category_regression_gate(current, "/nonexistent/path.json")
        self.assertTrue(gate["enabled"])
        self.assertIn("error", gate)
        self.assertFalse(gate["regression_detected"])

    def tearDown(self):
        # Clean up any temp files (best-effort).
        pass


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestAnchorConstants(unittest.TestCase):

    def test_preserved_33_count(self):
        self.assertEqual(len(PRESERVED_33_QUERY_IDS), 33)

    def test_subset_38_count(self):
        self.assertEqual(len(SUBSET_38_QUERY_IDS), 38)

    def test_33_is_subset_of_38(self):
        missing = set(PRESERVED_33_QUERY_IDS) - set(SUBSET_38_QUERY_IDS)
        self.assertEqual(missing, set(),
                         f"These PRESERVED_33 IDs are not in SUBSET_38: {missing}")

    def test_no_duplicates_in_33(self):
        self.assertEqual(len(PRESERVED_33_QUERY_IDS), len(set(PRESERVED_33_QUERY_IDS)))

    def test_no_duplicates_in_38(self):
        self.assertEqual(len(SUBSET_38_QUERY_IDS), len(set(SUBSET_38_QUERY_IDS)))


if __name__ == "__main__":
    unittest.main()
