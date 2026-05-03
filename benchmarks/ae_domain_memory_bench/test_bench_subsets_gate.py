"""Tests for S1c subset reporting + S1d comparable-label regression gate.

Run with:
    python3.11 -m unittest benchmarks.ae_domain_memory_bench.test_bench_subsets_gate -v
or from the bench dir:
    python3.11 -m unittest discover -s benchmarks/ae_domain_memory_bench -v

T12 2026-05-03: Extended with bench_meta_filter coverage (4 tests at the
bottom of the file). Pre-existing tests untouched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import the module under test.  Works whether run from repo root or bench dir.
import sys, os
_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from run_ae_domain_bench import (
    BENCH_META_EXCLUDE_CONTENT_PATTERNS,
    BENCH_META_EXCLUDE_IDS,
    PRESERVED_33_QUERY_IDS,
    SUBSET_38_QUERY_IDS,
    _bench_meta_filter_enabled,
    _build_subset,
    _build_subsets,
    _category_regression_gate,
    _is_bench_meta,
    run_scored,
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


# ---------------------------------------------------------------------------
# T12 — Bench-meta filter tests
#
# Per S6-DIAG (Sonnet packet 2026-05-03 read-only diagnosis), 8 claude_memory
# documents in substrate enumerate bench query IDs + GT memory_ids in their
# content and out-rank the actual GT memories. The filter implemented in
# run_ae_domain_bench.py excludes them at the bench-eval layer.
#
# These tests stub out NeuralMemory.hybrid_recall + DB content lookup so they
# are hermetic (no substrate dependency, no network). Pre-existing tests
# remain unaffected.
# ---------------------------------------------------------------------------


class _FakeMemory:
    """Minimal stand-in for NeuralMemory.hybrid_recall used by run_scored.

    Returns a fixed retrieval list per query (id-only matters for scoring).
    Honors the `k` argument so over-fetch behavior is testable.
    """

    def __init__(self, retrieval_per_query: dict[str, list[int]]):
        self._per_query = retrieval_per_query

    def hybrid_recall(self, query: str, k: int = 5, *, rerank=None,
                      mmr_lambda=0.0, percentile_floor=0.0,
                      **_kwargs) -> list[dict]:
        # Match by query_id encoded in the query string (test pattern).
        rids = self._per_query.get(query, [])
        return [{"id": rid} for rid in rids[:k]]


def _q(qid: str, query: str, gt: list[int],
       category: str = "electrical_contracting") -> dict:
    return {
        "id": qid,
        "category": category,
        "query": query,
        "ground_truth_ids": gt,
        "expected_channels": [],
    }


class TestBenchMetaFilter(unittest.TestCase):
    """T12 acceptance tests for the bench-meta exclusion filter."""

    def setUp(self):
        # Ensure env var is clean — tests that need it set use mock.patch.dict.
        os.environ.pop("NM_BENCH_DISABLE_META_FILTER", None)

    def tearDown(self):
        os.environ.pop("NM_BENCH_DISABLE_META_FILTER", None)

    def test_bench_meta_filter_excludes_known_meta_ids(self):
        """Filter must exclude every ID in BENCH_META_EXCLUDE_IDS by ID alone,
        independent of content. Real GT IDs in the same retrieval list must be
        preserved and surface into top-5 once meta IDs are filtered out.
        """
        # Confirm the curated list is non-empty and covers the S6-DIAG
        # findings (the high-frequency offender 7931 must be present).
        self.assertIn(7931, BENCH_META_EXCLUDE_IDS,
                      "S6-DIAG primary distractor 7931 must be curated")
        self.assertIn(7928, BENCH_META_EXCLUDE_IDS,
                      "S6-DIAG distractor 7928 must be curated")
        self.assertIn(14459, BENCH_META_EXCLUDE_IDS,
                      "S6-DIAG distractor 14459 must be curated")

        # ID-only check (content empty): every curated ID is filtered.
        for mid in BENCH_META_EXCLUDE_IDS:
            self.assertTrue(_is_bench_meta(mid, ""),
                            f"_is_bench_meta should exclude curated id {mid}")
            self.assertTrue(_is_bench_meta(mid, None),
                            f"_is_bench_meta should exclude curated id {mid} "
                            "even with None content")

        # End-to-end run_scored: bench-meta IDs in front, real GT IDs after.
        # k=5; without filter, GT id (999) sits at rank 6 and misses. With
        # filter, all 5 curated IDs drop out and GT moves into top-5.
        retrieval = {
            "LOT-015 query": [7928, 7931, 7171, 14459, 7975, 999, 1000, 1001, 1002, 1003],
        }
        queries = [_q("LOT-015", "LOT-015 query", [999], "lennar_lots")]
        # Mock _fetch_contents_for_ids to return empty (force ID-only path).
        with mock.patch(
            "run_ae_domain_bench._fetch_contents_for_ids",
            return_value={},
        ):
            mem = _FakeMemory(retrieval)
            result = run_scored(mem, queries, k=5, rerank=False)
        # GT 999 must hit at top-5 since 4+ meta IDs got filtered.
        rids = result["per_query"][0]["retrieved_ids"]
        self.assertEqual(result["per_query"][0]["hit_at_5"], 1,
                         f"GT must surface in top-5 after filter; got rids={rids}")
        self.assertNotIn(7931, rids, "filtered curated ID must not appear in result")
        self.assertNotIn(7928, rids, "filtered curated ID must not appear in result")

    def test_bench_meta_filter_does_not_exclude_real_gt(self):
        """Real ground-truth memory IDs (e.g., 274, 286, 277, 288, 5961, 4666)
        from S6-DIAG are NOT in the exclude list and must NOT be filtered when
        their content is genuine doctrine (no bench-meta patterns).
        """
        # Sample GT IDs from S6-DIAG §2 — these are the actual answer memories,
        # never the bench-meta docs.
        real_gt_samples = [274, 286, 277, 288, 5961, 4666, 13692, 13693,
                           14786, 14787, 6700, 7377, 6671, 7363]
        for mid in real_gt_samples:
            self.assertNotIn(
                mid, BENCH_META_EXCLUDE_IDS,
                f"GT id {mid} must NOT be in exclude list (S6-DIAG §2)",
            )

        # ID-only check: with empty content, real GT IDs pass.
        for mid in real_gt_samples:
            self.assertFalse(
                _is_bench_meta(mid, ""),
                f"_is_bench_meta must NOT exclude real GT id {mid} "
                "with empty content",
            )

        # Content-pattern check: representative GT content snippets must pass.
        # These are paraphrases of real GT content from S6-DIAG (no bench-meta
        # tokens like `ids=[`, `R@5`, `hit_at_5`, etc.).
        innocent_contents = [
            "Lennar lot 27 needs panel labels.",  # toy GT 274/286
            "GFCI exterior receptacles required.",  # GT 277/288
            "WA crew escalation: materials_missing — falta el wire de 12 gauge",
            "Amazon GFCI SKUs B0BW667149 / B0BW5V5N6L for outdoor receptacles",
            "OneDrive permits manifest: 1366 Braymore-Panel.pdf",
        ]
        for content in innocent_contents:
            self.assertFalse(
                _is_bench_meta(99999, content),
                f"non-curated id with innocent content must NOT be filtered: "
                f"{content[:60]!r}",
            )

        # End-to-end: pure-GT retrieval, no bench-meta IDs — every result
        # passes through and original ranking is preserved.
        retrieval = {
            "ELC-001 query": [277, 288, 5961, 999, 1000],
        }
        queries = [_q("ELC-001", "ELC-001 query", [277, 288])]
        with mock.patch(
            "run_ae_domain_bench._fetch_contents_for_ids",
            return_value={},
        ):
            mem = _FakeMemory(retrieval)
            result = run_scored(mem, queries, k=5, rerank=False)
        rids = result["per_query"][0]["retrieved_ids"]
        self.assertEqual(rids, [277, 288, 5961, 999, 1000],
                         "pure-GT retrieval must pass through unchanged")
        self.assertEqual(result["per_query"][0]["hit_at_5"], 1)

    def test_bench_meta_filter_provenance_recorded_in_artifact(self):
        """The scored-result artifact must contain a `bench_meta_filter` block
        with: enabled flag, curated ID list, content patterns, fetch_k, k,
        per-query exclusion counts/IDs, and disabled-via-env flag.
        """
        retrieval = {
            "LOT-015 query": [7931, 7928, 999, 1000, 1001],  # 2 meta filtered
            "ELC-001 query": [277, 288, 999, 1000, 1001],     # 0 meta filtered
        }
        queries = [
            _q("LOT-015", "LOT-015 query", [999], "lennar_lots"),
            _q("ELC-001", "ELC-001 query", [277, 288]),
        ]
        with mock.patch(
            "run_ae_domain_bench._fetch_contents_for_ids",
            return_value={},
        ):
            mem = _FakeMemory(retrieval)
            result = run_scored(mem, queries, k=5, rerank=False)

        self.assertIn("bench_meta_filter", result,
                      "scored artifact must contain bench_meta_filter block")
        bmf = result["bench_meta_filter"]

        # Required keys
        for key in (
            "enabled", "excluded_ids_curated", "exclude_content_patterns",
            "fetch_k", "k", "queries_with_exclusions",
            "excluded_count_per_query", "excluded_ids_per_query",
            "disabled_via_env",
        ):
            self.assertIn(key, bmf, f"bench_meta_filter missing key: {key}")

        # Filter is on by default → enabled True, disabled_via_env False.
        self.assertTrue(bmf["enabled"], "filter must be ON by default")
        self.assertFalse(bmf["disabled_via_env"])

        # Curated list matches module constant.
        self.assertEqual(bmf["excluded_ids_curated"], list(BENCH_META_EXCLUDE_IDS))
        self.assertEqual(
            bmf["exclude_content_patterns"],
            list(BENCH_META_EXCLUDE_CONTENT_PATTERNS),
        )

        # k vs fetch_k — over-fetch by len(BENCH_META_EXCLUDE_IDS).
        self.assertEqual(bmf["k"], 5)
        self.assertEqual(bmf["fetch_k"], 5 + len(BENCH_META_EXCLUDE_IDS))

        # Per-query exclusion record: only LOT-015 had meta IDs.
        self.assertEqual(bmf["queries_with_exclusions"], 1)
        self.assertEqual(bmf["excluded_count_per_query"], {"LOT-015": 2})
        self.assertEqual(
            sorted(bmf["excluded_ids_per_query"]["LOT-015"]),
            sorted([7931, 7928]),
        )
        self.assertNotIn("ELC-001", bmf["excluded_ids_per_query"])

    def test_bench_meta_filter_disabled_via_env_var(self):
        """Setting NM_BENCH_DISABLE_META_FILTER=1 must turn the filter off:
          - bench-meta IDs are NOT removed from retrieved_ids
          - bench_meta_filter.enabled = False
          - bench_meta_filter.disabled_via_env = True
          - fetch_k = k (no over-fetch)
        Provides a reproducible escape hatch for sanity checks vs prior
        artifacts.
        """
        retrieval = {
            "LOT-015 query": [7931, 7928, 7171, 14459, 7975, 999],
        }
        queries = [_q("LOT-015", "LOT-015 query", [999], "lennar_lots")]
        with mock.patch.dict(os.environ, {"NM_BENCH_DISABLE_META_FILTER": "1"}):
            self.assertFalse(_bench_meta_filter_enabled(),
                             "env var must disable filter")
            with mock.patch(
                "run_ae_domain_bench._fetch_contents_for_ids",
                return_value={},
            ):
                mem = _FakeMemory(retrieval)
                result = run_scored(mem, queries, k=5, rerank=False)

        # Filter disabled → curated meta IDs survive in top-5.
        rids = result["per_query"][0]["retrieved_ids"]
        self.assertIn(7931, rids,
                      "with filter off, curated bench-meta ID 7931 must survive")
        self.assertIn(7928, rids,
                      "with filter off, curated bench-meta ID 7928 must survive")
        self.assertEqual(result["per_query"][0]["hit_at_5"], 0,
                         "with filter off, GT 999 stays out of top-5")

        bmf = result["bench_meta_filter"]
        self.assertFalse(bmf["enabled"])
        self.assertTrue(bmf["disabled_via_env"])
        self.assertEqual(bmf["fetch_k"], 5,
                         "no over-fetch when filter is disabled")
        self.assertEqual(bmf["queries_with_exclusions"], 0)
        self.assertEqual(bmf["excluded_count_per_query"], {})

    # ---- defensive coverage: pattern filter + edge cases ------------------

    def test_pattern_filter_catches_uncurated_bench_meta(self):
        """Defense-in-depth: a NEW bench-meta doc that lands in substrate
        AFTER the curated list was frozen must still be caught by content
        pattern matching, not just ID-list membership.
        """
        # New ID, never seen in curated list.
        new_id = 99001
        self.assertNotIn(new_id, BENCH_META_EXCLUDE_IDS)
        bench_meta_content_samples = [
            "ELC-027 ids=[274, 286] panel labels seed",
            "Bench R@5=0.65 after dedupe pass",
            "hit_at_5 column shows 0 for FIN-002 due to label mismatch",
            "GT memory_ids 277, 288 are the doctrine pair",
        ]
        for content in bench_meta_content_samples:
            self.assertTrue(
                _is_bench_meta(new_id, content),
                f"pattern filter must catch uncurated bench-meta: "
                f"{content[:60]!r}",
            )


# ---------------------------------------------------------------------------
# S1h — Top-level threshold_failed + regression_detected boolean contracts
#
# The actual emission of `threshold_failed` and `regression_detected` happens
# in main() after run_scored() and _category_regression_gate() return.  These
# unit tests verify the upstream results those fields are derived from, making
# the one-liner derivation in main() self-evidently correct.
# ---------------------------------------------------------------------------

class TestS1hThresholdBooleans(unittest.TestCase):
    """S1h acceptance tests for the threshold_failed / regression_detected
    boolean contracts introduced in 2026-05-03."""

    def test_categories_failed_non_empty_on_miss(self):
        """categories_failed must be non-empty when a category falls below
        threshold → threshold_failed = bool(categories_failed) = True."""
        queries = [_q("ELC-001", "ELC-001 query", [999])]
        with mock.patch("run_ae_domain_bench._fetch_contents_for_ids", return_value={}):
            result = run_scored(_FakeMemory({"ELC-001 query": []}), queries, k=5, rerank=False)
        self.assertTrue(
            bool(result.get("categories_failed")),
            "categories_failed must be non-empty on miss → threshold_failed=True",
        )

    def test_categories_failed_empty_on_pass(self):
        """categories_failed must be empty when all categories pass their
        threshold → threshold_failed = False."""
        queries = [_q("ELC-001", "ELC-001 query", [999])]
        with mock.patch("run_ae_domain_bench._fetch_contents_for_ids", return_value={}):
            result = run_scored(_FakeMemory({"ELC-001 query": [999]}), queries, k=5, rerank=False)
        self.assertFalse(
            bool(result.get("categories_failed")),
            "categories_failed must be empty when all categories pass → threshold_failed=False",
        )

    def test_regression_detected_fires_on_drop(self):
        """category_regression_gate.regression_detected must be True when a
        category drops > 0.05 R@5 on the comparable intersect → this field is
        hoisted to top-level regression_detected in main()."""
        ids = ["ELC-001"]
        prev_rows = [_make_pq_row(qid, "electrical_contracting", h5=1) for qid in ids]
        cur_rows  = [_make_pq_row(qid, "electrical_contracting", h5=0) for qid in ids]
        prev_path = _write_artifact({
            "mode": "scored",
            "per_query": prev_rows,
            "per_category": {},
            "category_regression_gate": {
                "enabled": False, "regressions": [], "regression_detected": False,
            },
        })
        current = _make_scored_result(cur_rows)
        gate = _category_regression_gate(current, prev_path)
        self.assertTrue(
            gate["regression_detected"],
            "regression_detected must be True on > 0.05 R@5 drop in intersect",
        )

    def test_regression_detected_false_without_prev(self):
        """Without a --prev-results artifact the gate is disabled and
        regression_detected must be False."""
        current = _make_scored_result([_make_pq_row("ELC-001", "electrical_contracting")])
        gate = _category_regression_gate(current, None)
        self.assertFalse(
            gate["regression_detected"],
            "regression_detected must be False when no prior artifact is given",
        )


if __name__ == "__main__":
    unittest.main()
