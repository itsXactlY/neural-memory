"""Acceptance tests for Sprint 2 Phase 7 Commit 3 — entity extraction.

Per addendum lines 218-262. Split across two surfaces:

Extraction (unit, no DB):
  - extract_entities returns capitalized words, filtered by stopwords
  - case-insensitive deduplication within a single text
  - empty text returns empty list

Registry (unit, sqlite3 only — no embedder):
  - get_or_create_entity creates kind='entity' node on first call
  - second call with same name (any case) reuses existing entity
  - frequency increments on reuse
  - get_entity returns dict with frequency
  - count_entities_named is case-insensitive
  - link_memory_to_entity creates mentions_entity edge
  - get_entities_for_memory returns linked entities
  - process_memory end-to-end: extract → create → link

Stdlib unittest. Run:
    python3 python/test_entity_extraction.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entity_extraction import EntityRegistry, extract_entities  # noqa: E402
from memory_client import SQLiteStore  # noqa: E402


class ExtractEntitiesTests(unittest.TestCase):
    def test_extracts_capitalized_proper_nouns(self) -> None:
        result = extract_entities("Sarah from Lennar asked about lot 27.")
        self.assertIn("Sarah", result)
        self.assertIn("Lennar", result)

    def test_filters_sentence_initial_stopwords(self) -> None:
        # "When" and "If" are stopwords even though capitalized
        result = extract_entities("When estimating panels, check load calc.")
        self.assertNotIn("When", result)

    def test_dedupes_within_single_text(self) -> None:
        result = extract_entities("Lennar lot 12. Lennar lot 13.")
        self.assertEqual(result.count("Lennar"), 1)

    def test_empty_text_returns_empty_list(self) -> None:
        self.assertEqual(extract_entities(""), [])
        self.assertEqual(extract_entities("   "), [])

    def test_lowercase_text_returns_no_entities(self) -> None:
        # v1 limitation; case-insensitive resolution happens at registry level
        result = extract_entities("lennar asked about photos.")
        self.assertEqual(result, [])

    def test_acronyms_pass_through(self) -> None:
        # NEC, GFCI etc. are valid entities for AE retrieval
        result = extract_entities("NEC requires GFCI on outdoor receptacles.")
        self.assertIn("NEC", result)
        self.assertIn("GFCI", result)


class EntityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(str(Path(self._tmp.name) / "memory.db"))
        self.registry = EntityRegistry(self.store)

    def tearDown(self) -> None:
        try:
            self.store.conn.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_get_or_create_creates_new_entity(self) -> None:
        eid = self.registry.get_or_create_entity("Lennar")
        self.assertIsInstance(eid, int)
        # Verify row exists with kind='entity'
        row = self.store.conn.execute(
            "SELECT label, kind, origin_system FROM memories WHERE id = ?",
            (eid,)
        ).fetchone()
        self.assertEqual(row[0], "Lennar")
        self.assertEqual(row[1], "entity")
        self.assertEqual(row[2], "entity_extractor")

    def test_get_or_create_dedupes_case_insensitively(self) -> None:
        eid1 = self.registry.get_or_create_entity("Lennar")
        eid2 = self.registry.get_or_create_entity("LENNAR")
        eid3 = self.registry.get_or_create_entity("lennar")
        self.assertEqual(eid1, eid2)
        self.assertEqual(eid1, eid3)
        self.assertEqual(self.registry.count_entities_named("Lennar"), 1)

    def test_frequency_increments_on_reuse(self) -> None:
        eid = self.registry.get_or_create_entity("Lennar")
        self.registry.get_or_create_entity("Lennar")
        self.registry.get_or_create_entity("lennar")
        ent = self.registry.get_entity("Lennar")
        # First create initialized frequency to 1, two more touches = 3
        self.assertEqual(ent["frequency"], 3)

    def test_get_entity_returns_none_for_unknown(self) -> None:
        self.assertIsNone(self.registry.get_entity("NobodyHome"))

    def test_count_entities_named_case_insensitive(self) -> None:
        self.registry.get_or_create_entity("Lennar")
        self.assertEqual(self.registry.count_entities_named("Lennar"), 1)
        self.assertEqual(self.registry.count_entities_named("lennar"), 1)
        self.assertEqual(self.registry.count_entities_named("LENNAR"), 1)
        self.assertEqual(self.registry.count_entities_named("Pulte"), 0)

    def test_link_memory_to_entity_creates_typed_edge(self) -> None:
        # Create a non-entity memory row
        mid = self.store.store("test", "Lennar lot 27", [0.1, 0.2])
        eid = self.registry.get_or_create_entity("Lennar")
        self.registry.link_memory_to_entity(mid, eid)
        # Verify edge with edge_type='mentions_entity'
        edges = self.store.conn.execute(
            "SELECT edge_type FROM connections WHERE source_id = ? AND target_id = ?",
            (mid, eid)
        ).fetchall()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][0], "mentions_entity")

    def test_get_entities_for_memory_returns_linked(self) -> None:
        mid = self.store.store("m", "Sarah from Lennar", [0.1, 0.2])
        sid = self.registry.get_or_create_entity("Sarah")
        lid = self.registry.get_or_create_entity("Lennar")
        self.registry.link_memory_to_entity(mid, sid)
        self.registry.link_memory_to_entity(mid, lid)

        ents = self.registry.get_entities_for_memory(mid)
        labels = sorted(e["label"] for e in ents)
        self.assertEqual(labels, ["Lennar", "Sarah"])

    def test_process_memory_end_to_end(self) -> None:
        mid = self.store.store("m", "Sarah from Lennar asked about lot 27.", [0.1, 0.2])
        ids = self.registry.process_memory(mid, "Sarah from Lennar asked about lot 27.")
        self.assertEqual(len(ids), 2)

        ents = self.registry.get_entities_for_memory(mid)
        labels = sorted(e["label"] for e in ents)
        self.assertEqual(labels, ["Lennar", "Sarah"])

    def test_process_memory_increments_existing_frequency(self) -> None:
        mid1 = self.store.store("m1", "Lennar lot 12 needs panel labels.", [0.1, 0.2])
        mid2 = self.store.store("m2", "Lennar lot 13 needs GFCI correction.", [0.1, 0.2])
        self.registry.process_memory(mid1, "Lennar lot 12 needs panel labels.")
        self.registry.process_memory(mid2, "Lennar lot 13 needs GFCI correction.")
        lennar = self.registry.get_entity("Lennar")
        # Created via memory 1, touched via memory 2
        self.assertGreaterEqual(lennar["frequency"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
