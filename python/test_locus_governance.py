"""Acceptance tests for Sprint 2 Phase 7 Commit 10 — locus overlay,
governance, and explanation paths.

Per addendum lines 511-565. Eight contracts:

  1. create_locus produces locus nodes; assign_locus adds located_in edge
  2. Locus overlay does NOT replace graph (kind='locus' nodes coexist with
     other memories on the same memories table)
  3. memory_count excludes overlay nodes (entities + loci) by default
  4. explain_recall returns results with 'explanation' dict containing
     channels, final_score, and features (including salience)
  5. forget(mode='background') sets memory_visibility='backgrounded'
  6. forget(mode='redact') replaces content with redaction marker
  7. forget(mode='delete') removes the row
  8. forget rejects unknown modes

Stdlib unittest. Run:
    python3 python/test_locus_governance.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class LocusOverlayTests(unittest.TestCase):
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

    def test_create_locus_and_assign(self) -> None:
        room = self.mem.create_locus("Business Ops", "Compliance Room")
        mid = self.mem.remember("Permit follow-up due Friday.", detect_conflicts=False)
        self.mem.assign_locus(mid, room)

        self.assertTrue(self.mem.has_edge(mid, room, edge_type="located_in"))

        # Verify the locus is a kind='locus' row
        row = self.mem.get_memory(room)
        self.assertEqual(row["kind"], "locus")
        self.assertEqual(row["label"], "Compliance Room")

    def test_locus_overlay_does_not_replace_graph(self) -> None:
        # Memories and loci coexist on the same memories table
        self.mem.create_locus("Business Ops", "Compliance Room")
        self.mem.remember("Permit follow-up due Friday.", detect_conflicts=False)
        self.assertGreaterEqual(self.mem.memory_count(), 1)
        # full count includes loci + entities
        full = self.mem.memory_count(exclude_overlay=False)
        user = self.mem.memory_count(exclude_overlay=True)
        self.assertGreater(full, user)

    def test_assign_locus_is_idempotent(self) -> None:
        room = self.mem.create_locus("Wing", "Room")
        mid = self.mem.remember("Memo", detect_conflicts=False)
        self.mem.assign_locus(mid, room)
        self.mem.assign_locus(mid, room)  # second call is no-op
        # Count located_in edges between mid and room
        rows = self.mem.store.conn.execute(
            "SELECT COUNT(*) FROM connections "
            "WHERE source_id = ? AND target_id = ? AND edge_type = 'located_in'",
            (mid, room),
        ).fetchone()
        self.assertEqual(rows[0], 1, "assign_locus must be idempotent")

    def test_create_locus_dedupes_existing(self) -> None:
        a = self.mem.create_locus("Wing", "Room")
        b = self.mem.create_locus("Wing", "Room")
        self.assertEqual(a, b)


class ExplanationPathTests(unittest.TestCase):
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

    def test_explain_recall_returns_explanation(self) -> None:
        mid = self.mem.remember("Sarah is the Lennar contact.",
                                detect_conflicts=False, kind="experience")
        results = self.mem.explain_recall("Who is the Lennar contact?", k=5)
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("explanation", r)
        self.assertIn("channels", r["explanation"])
        self.assertIn("final_score", r["explanation"])
        self.assertIn("salience", r["explanation"]["features"])


class GovernanceTests(unittest.TestCase):
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

    def test_forget_background_sets_visibility(self) -> None:
        mid = self.mem.remember("Temporary note.", detect_conflicts=False)
        self.mem.forget(mid, mode="background")
        row = self.mem.get_memory(mid)
        self.assertEqual(row["memory_visibility"], "backgrounded")

    def test_forget_does_not_break_edges_in_background_mode(self) -> None:
        m1 = self.mem.remember("Memory A.", detect_conflicts=False)
        m2 = self.mem.remember("Memory B.", detect_conflicts=False)
        self.mem.store.add_connection(m1, m2, weight=1.0, edge_type="similar")

        self.mem.forget(m1, mode="background")
        # Edge still exists
        self.assertTrue(self.mem.has_edge(m1, m2))

    def test_forget_redact_replaces_content(self) -> None:
        mid = self.mem.remember("Sensitive content.", detect_conflicts=False)
        self.mem.forget(mid, mode="redact")
        row = self.mem.get_memory(mid)
        self.assertEqual(row["content"], "[REDACTED]")
        self.assertEqual(row["memory_visibility"], "hidden")

    def test_forget_delete_removes_row(self) -> None:
        mid = self.mem.remember("Doomed memory.", detect_conflicts=False)
        self.mem.forget(mid, mode="delete")
        self.assertIsNone(self.mem.get_memory(mid))

    def test_forget_unknown_mode_raises(self) -> None:
        mid = self.mem.remember("test", detect_conflicts=False)
        with self.assertRaises(ValueError):
            self.mem.forget(mid, mode="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
