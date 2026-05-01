"""Acceptance tests for Sprint 2 Phase 7 Commit 2 — retain-time memory typing.

Per addendum lines 159-220. Six contracts split across two surfaces:

Classifier (unit, no DB):
  1. defaults to experience for event-shaped text
  2. classifies "When..." how-to text as procedural
  3. classifies NEC/code requirements as world
  4. classifies summary/inference text as mental_model
  5. metadata kind override beats heuristics

Store layer (unit, sqlite3 only — no embedder):
  6. SQLiteStore.store accepts new kwargs and persists them
  7. SQLiteStore.store backward-compat: 3-arg positional call still works
  8. SQLiteStore.store auto-stamps transaction_time when not provided

Stdlib unittest. Run:
    python3 python/test_memory_typing.py
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_memory_kind import classify_memory_kind  # noqa: E402
from memory_client import SQLiteStore  # noqa: E402
from memory_types import MEMORY_KINDS  # noqa: E402


class ClassifyMemoryKindTests(unittest.TestCase):
    """Unit tests for the heuristic classifier."""

    def test_event_text_defaults_to_experience(self) -> None:
        self.assertEqual(
            classify_memory_kind("Customer asked about panel upgrade on Friday."),
            "experience",
        )

    def test_how_to_classifies_as_procedural(self) -> None:
        self.assertEqual(
            classify_memory_kind("When estimating a panel upgrade, check load calc first."),
            "procedural",
        )
        self.assertEqual(
            classify_memory_kind("If permit jurisdiction is unclear, call the village before quoting."),
            "procedural",
        )
        self.assertEqual(
            classify_memory_kind("Always photograph panel labels before leaving the job."),
            "procedural",
        )

    def test_nec_or_code_classifies_as_world(self) -> None:
        self.assertEqual(
            classify_memory_kind("The NEC requires GFCI on outdoor receptacles."),
            "world",
        )
        self.assertEqual(
            classify_memory_kind("Chicago code mandates conduit for residential branch circuits."),
            "world",
        )

    def test_inference_classifies_as_mental_model(self) -> None:
        self.assertEqual(
            classify_memory_kind("We've concluded that Lennar prefers EOM invoicing."),
            "mental_model",
        )
        self.assertEqual(
            classify_memory_kind("It seems Vibha's timeline is driven by HOA approval."),
            "mental_model",
        )

    def test_metadata_kind_override_beats_heuristics(self) -> None:
        # Heuristic would say procedural; metadata says world. metadata wins.
        self.assertEqual(
            classify_memory_kind(
                "When the inspection happens, the NEC requires GFCI.",
                metadata={"kind": "world"},
            ),
            "world",
        )

    def test_metadata_kind_invalid_falls_back_to_heuristic(self) -> None:
        # Invalid metadata kind should NOT win; heuristic kicks in.
        self.assertEqual(
            classify_memory_kind(
                "Customer asked about panel upgrade.",
                metadata={"kind": "totally_made_up"},
            ),
            "experience",
        )

    def test_returned_kind_is_always_in_memory_kinds(self) -> None:
        samples = [
            "Random text",
            "",
            "12345",
            "When the time is right, do the thing.",
            "The NEC says foo.",
            "It seems X.",
        ]
        for text in samples:
            kind = classify_memory_kind(text)
            self.assertIn(kind, MEMORY_KINDS)

    def test_empty_text_returns_unknown(self) -> None:
        self.assertEqual(classify_memory_kind(""), "unknown")
        self.assertEqual(classify_memory_kind("   "), "unknown")


class SQLiteStoreTypedKwargsTests(unittest.TestCase):
    """Unit tests for SQLiteStore.store() with typed kwargs.

    SQLiteStore.__init__ now invokes SchemaUpgrade automatically, so a fresh
    tmp DB has the Phase 7 columns ready.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.store = SQLiteStore(self.db_path)
        self.embedding = [0.1, 0.2, 0.3, 0.4]

    def tearDown(self) -> None:
        try:
            self.store.conn.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def _read_row(self, mem_id: int) -> sqlite3.Row:
        self.store.conn.row_factory = sqlite3.Row
        return self.store.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()

    def test_backward_compat_3_arg_positional(self) -> None:
        """Existing call shape must continue to work unchanged."""
        mid = self.store.store("Pet", "Dog named Lou", self.embedding)
        row = self._read_row(mid)
        self.assertEqual(row["label"], "Pet")
        self.assertEqual(row["content"], "Dog named Lou")
        # transaction_time auto-stamped even on positional call
        self.assertIsNotNone(row["transaction_time"])

    def test_typed_kwargs_persist_correctly(self) -> None:
        before = time.time()
        mid = self.store.store(
            "Procedural rule",
            "When estimating panel upgrades, check load calc first.",
            self.embedding,
            kind="procedural",
            confidence=0.95,
            source="estimator_workflow",
            origin_system="ae",
            valid_from=1000.0,
            metadata={"crew": "spanish", "site": "lennar-27"},
        )
        after = time.time()

        row = self._read_row(mid)
        self.assertEqual(row["kind"], "procedural")
        self.assertAlmostEqual(row["confidence"], 0.95)
        self.assertEqual(row["source"], "estimator_workflow")
        self.assertEqual(row["origin_system"], "ae")
        self.assertEqual(row["valid_from"], 1000.0)
        self.assertGreaterEqual(row["transaction_time"], before)
        self.assertLessEqual(row["transaction_time"], after)
        meta = json.loads(row["metadata_json"])
        self.assertEqual(meta["crew"], "spanish")
        self.assertEqual(meta["site"], "lennar-27")

    def test_explicit_transaction_time_preserved(self) -> None:
        mid = self.store.store(
            "Bi-temporal probe",
            "Sarah was the contact in March.",
            self.embedding,
            valid_from=100.0,
            valid_to=200.0,
            transaction_time=150.0,
        )
        row = self._read_row(mid)
        self.assertEqual(row["transaction_time"], 150.0)
        self.assertEqual(row["valid_from"], 100.0)
        self.assertEqual(row["valid_to"], 200.0)

    def test_empty_metadata_dict_yields_null_metadata_json(self) -> None:
        # `metadata=None` and `metadata={}` both should not write a JSON value.
        mid_none = self.store.store("a", "b", self.embedding, metadata=None)
        mid_empty = self.store.store("a", "b", self.embedding, metadata={})
        row_none = self._read_row(mid_none)
        row_empty = self._read_row(mid_empty)
        self.assertIsNone(row_none["metadata_json"])
        self.assertIsNone(row_empty["metadata_json"])

    def test_schema_upgrade_runs_on_fresh_init(self) -> None:
        """Fresh SQLiteStore on empty path must already have Phase 7 columns."""
        cols = {row[1] for row in self.store.conn.execute("PRAGMA table_info(memories)")}
        for required in ("kind", "confidence", "transaction_time",
                         "origin_system", "metadata_json", "memory_visibility"):
            self.assertIn(required, cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
