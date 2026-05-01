"""Acceptance tests for Sprint 2 Phase 7 Commit 5 — sparse + temporal channels.

Per addendum lines 300-336. Five contracts:

  1. sparse_search retrieves exact jargon via FTS5
  2. temporal_search prefers valid record at as_of (current query)
  3. temporal_search returns old fact when as_of predates supersession
  4. sparse_search works on fresh install (FTS5 backfill via schema_upgrade)
  5. temporal_search treats NULL validity as always-valid

Stdlib unittest. Run:
    python3 python/test_sparse_temporal.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402
from schema_upgrade import SchemaUpgrade  # noqa: E402


class SparseSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_sparse_finds_exact_jargon(self) -> None:
        mid = self.mem.remember(
            "The job needs GFCI protection on exterior receptacles.",
            detect_conflicts=False,
        )
        # Add some unrelated memories
        self.mem.remember("Customer asked about lighting.", detect_conflicts=False)
        self.mem.remember("Schedule the inspection.", detect_conflicts=False)

        results = self.mem.sparse_search("GFCI exterior receptacles")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]['id'], mid)

    def test_sparse_search_respects_k_limit(self) -> None:
        for i in range(20):
            self.mem.remember(f"Job {i} mentions panel upgrade work.", detect_conflicts=False)
        results = self.mem.sparse_search("panel upgrade", k=5)
        self.assertLessEqual(len(results), 5)

    def test_sparse_search_returns_empty_for_no_match(self) -> None:
        self.mem.remember("Real content here.", detect_conflicts=False)
        results = self.mem.sparse_search("xyzabc_no_match_token")
        self.assertEqual(results, [])

    def test_sparse_search_works_on_fresh_install(self) -> None:
        """Fresh tmp DB created via SchemaUpgrade must have FTS5 ready."""
        # NeuralMemory.__init__ already invokes SchemaUpgrade; verify FTS5 table
        # exists by querying sqlite_master.
        row = self.mem.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        self.assertIsNotNone(row, "memories_fts virtual table not created on fresh install")

    def test_sparse_search_handles_empty_query(self) -> None:
        self.mem.remember("Some content.", detect_conflicts=False)
        self.assertEqual(self.mem.sparse_search(""), [])
        self.assertEqual(self.mem.sparse_search("   "), [])


class TemporalSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_prefers_valid_record_at_as_of(self) -> None:
        old = self.mem.remember(
            "Sarah is not the Lennar contact.",
            detect_conflicts=False,
            valid_from=100.0,
            valid_to=200.0,
        )
        new = self.mem.remember(
            "Sarah is the Lennar contact.",
            detect_conflicts=False,
            valid_from=201.0,
        )

        results = self.mem.temporal_search("Lennar contact Sarah", as_of=300.0)
        ids = [r['id'] for r in results]
        # At as_of=300, old (valid 100-200) is invalid; new (valid 201-) is valid
        self.assertIn(new, ids)
        self.assertNotIn(old, ids)

    def test_returns_old_fact_for_past_query(self) -> None:
        old = self.mem.remember(
            "Miguel handled lot 27.",
            detect_conflicts=False,
            valid_from=100.0,
            valid_to=200.0,
        )
        new = self.mem.remember(
            "Sarah handled lot 27.",
            detect_conflicts=False,
            valid_from=201.0,
        )

        results = self.mem.temporal_search("who handled lot 27", as_of=150.0)
        ids = [r['id'] for r in results]
        # At as_of=150, old (100-200) is valid; new (201-) not yet valid
        self.assertIn(old, ids)
        self.assertNotIn(new, ids)

    def test_null_validity_is_always_valid(self) -> None:
        # No valid_from/valid_to: should be returned regardless of as_of
        m = self.mem.remember("Always-valid fact about panels.", detect_conflicts=False)
        results = self.mem.temporal_search("panels", as_of=999_999.0)
        self.assertIn(m, [r['id'] for r in results])

    def test_open_ended_validity_persists_into_future(self) -> None:
        # valid_from set, valid_to None: valid forever after valid_from
        m = self.mem.remember(
            "New process starting now.",
            detect_conflicts=False,
            valid_from=500.0,
        )
        # before valid_from
        before = self.mem.temporal_search("process", as_of=400.0)
        self.assertNotIn(m, [r['id'] for r in before])
        # after valid_from
        after = self.mem.temporal_search("process", as_of=10_000.0)
        self.assertIn(m, [r['id'] for r in after])


if __name__ == "__main__":
    unittest.main(verbosity=2)
