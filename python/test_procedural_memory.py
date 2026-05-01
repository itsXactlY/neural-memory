"""Acceptance tests for Sprint 2 Phase 7 Commit 4 — procedural memory retrieval
+ evidence-link edges.

Per addendum lines 263-298. Five contracts:

  1. recall(kind='procedural') returns only procedural-kind memories
  2. recall() with no kind kwarg behaves identically to pre-Phase-7
  3. evidence_ids creates derived_from edges from new memory to evidence
  4. procedural memories still appear in general (unfiltered) recall
  5. invalid evidence id does not block memory storage

Stdlib unittest. Run:
    python3 python/test_procedural_memory.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class ProceduralRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        # use_cpp=False forces Python paths (HNSW or brute-force) — works
        # without libneural_memory.so. embedding_backend='hash' is fast +
        # deterministic for test repeatability.
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

    # ----- Contract 1: kind filter returns only matching kind ----------------

    def test_kind_filter_returns_only_procedural(self) -> None:
        self.mem.remember("Customer asked about panel upgrade.", detect_conflicts=False)
        proc_id = self.mem.remember(
            "When estimating panel upgrades, check load calc first.",
            detect_conflicts=False,
        )
        self.mem.remember("The NEC requires GFCI on outdoor receptacles.", detect_conflicts=False)

        results = self.mem.recall("panel upgrade load calc", k=5, kind="procedural")

        self.assertGreater(len(results), 0, "expected at least 1 procedural result")
        result_ids = [r['id'] for r in results]
        self.assertIn(proc_id, result_ids)
        # Verify all returned results have kind='procedural'
        for r in results:
            row = self.mem.store.conn.execute(
                "SELECT kind FROM memories WHERE id = ?", (r['id'],)
            ).fetchone()
            self.assertEqual(row[0], "procedural", f"non-procedural memory in filtered result: id={r['id']}")

    # ----- Contract 2: default behavior unchanged when kind not specified ----

    def test_default_recall_unchanged(self) -> None:
        ids = []
        ids.append(self.mem.remember("Customer asked about panel upgrade.", detect_conflicts=False))
        ids.append(self.mem.remember("The NEC requires GFCI on outdoor receptacles.", detect_conflicts=False))

        # Without kind kwarg: original behavior
        unfiltered = self.mem.recall("panel upgrade", k=5)
        # Should return SOMETHING (one of our memories) and not crash
        self.assertGreater(len(unfiltered), 0)

    # ----- Contract 3: evidence_ids creates derived_from edges ---------------

    def test_evidence_ids_create_derived_from_edges(self) -> None:
        exp = self.mem.remember(
            "On the Oak job, missing permit jurisdiction caused delay.",
            detect_conflicts=False,
        )
        proc = self.mem.remember(
            "When scheduling panel upgrades, verify permit jurisdiction.",
            detect_conflicts=False,
            evidence_ids=[exp],
        )

        edges = self.mem.store.conn.execute(
            "SELECT source_id, target_id, edge_type FROM connections "
            "WHERE source_id = ? AND target_id = ? AND edge_type = 'derived_from'",
            (proc, exp),
        ).fetchall()
        self.assertEqual(len(edges), 1, f"expected 1 derived_from edge from {proc}->{exp}, got {len(edges)}")

    def test_evidence_ids_create_multiple_edges(self) -> None:
        e1 = self.mem.remember("Job A had permit delay.", detect_conflicts=False)
        e2 = self.mem.remember("Job B had permit delay.", detect_conflicts=False)
        proc = self.mem.remember(
            "When scheduling, always verify permits first.",
            detect_conflicts=False,
            evidence_ids=[e1, e2],
        )

        rows = self.mem.store.conn.execute(
            "SELECT target_id FROM connections "
            "WHERE source_id = ? AND edge_type = 'derived_from'",
            (proc,),
        ).fetchall()
        targets = {r[0] for r in rows}
        self.assertEqual(targets, {e1, e2})

    # ----- Contract 4: procedural memory in general recall -------------------

    def test_procedural_in_general_recall(self) -> None:
        proc = self.mem.remember(
            "When labeling panels, photograph before and after.",
            detect_conflicts=False,
        )

        results = self.mem.recall("panel photograph before after", k=10)
        self.assertIn(proc, [r['id'] for r in results])

    # ----- Contract 5: invalid evidence id does not block storage -----------

    def test_invalid_evidence_id_does_not_block_storage(self) -> None:
        # Pass a non-existent ID; remember should still succeed
        # The FOREIGN KEY constraint exists but SQLite doesn't enforce by default
        mid = self.mem.remember(
            "When dealing with weird stuff, try X.",
            detect_conflicts=False,
            evidence_ids=[999_999_999],
        )
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

    # ----- Contract 6: empty/None evidence_ids is a no-op --------------------

    def test_empty_evidence_ids_is_noop(self) -> None:
        mid_none = self.mem.remember("Plain memory.", detect_conflicts=False, evidence_ids=None)
        mid_empty = self.mem.remember("Another plain memory.", detect_conflicts=False, evidence_ids=[])

        edges_none = self.mem.store.conn.execute(
            "SELECT COUNT(*) FROM connections WHERE source_id = ? AND edge_type = 'derived_from'",
            (mid_none,),
        ).fetchone()[0]
        edges_empty = self.mem.store.conn.execute(
            "SELECT COUNT(*) FROM connections WHERE source_id = ? AND edge_type = 'derived_from'",
            (mid_empty,),
        ).fetchone()[0]

        self.assertEqual(edges_none, 0)
        self.assertEqual(edges_empty, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
