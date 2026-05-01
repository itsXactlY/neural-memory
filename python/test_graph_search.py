"""Acceptance tests for Sprint 2 Phase 7 Commit 7 — PPR + relation views.

Per addendum lines 376-418. Five contracts:

  1. graph_search reaches 2-hop related memory via mentions_entity + applies_to
  2. relation_view filter operates over a single connections table (no fork)
  3. entity-heavy query weights mentions_entity higher than semantic_similar_to
  4. available_relation_views returns the 5 expected views
  5. intent classifier maps "Who" / "When" / "Why" / "How" appropriately

Stdlib unittest. Run:
    python3 python/test_graph_search.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class GraphSearchTests(unittest.TestCase):
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

    def test_two_hop_related_memory_is_reachable(self) -> None:
        a = self.mem.remember(
            "Lennar lot 27 has inspection issue.",
            kind="experience",
            detect_conflicts=False,
        )
        b = self.mem.remember(
            "Lennar is a builder customer.",
            kind="entity",
            detect_conflicts=False,
        )
        c = self.mem.remember(
            "Inspection delays usually require photo evidence.",
            kind="procedural",
            detect_conflicts=False,
        )
        # Build the chain a -> b -> c
        self.mem.store.add_connection(a, b, weight=1.0, edge_type="mentions_entity")
        self.mem.store.add_connection(b, c, weight=1.0, edge_type="applies_to")

        results = self.mem.graph_search("Lennar inspection issue", k=5, hops=2)
        result_ids = [r['id'] for r in results]
        self.assertIn(c, result_ids,
                      f"expected c={c} reachable from a via 2-hop traversal; got {result_ids}")

    def test_relation_view_filter_uses_single_graph(self) -> None:
        self.mem.remember("A caused B.", kind="claim", detect_conflicts=False)
        views = self.mem.available_relation_views()
        self.assertIn("causal", views)
        self.assertTrue(self.mem.uses_single_connection_table())

    def test_entity_query_weights_entity_edges_higher_than_semantic(self) -> None:
        weights = self.mem.intent_edge_weights("Who is the Lennar contact?")
        self.assertIn("mentions_entity", weights)
        self.assertIn("semantic_similar_to", weights)
        self.assertGreaterEqual(weights["mentions_entity"], weights["semantic_similar_to"])

    def test_temporal_query_weights_happened_before_higher(self) -> None:
        weights = self.mem.intent_edge_weights("When did the inspection happen?")
        self.assertGreaterEqual(weights["happened_before"], weights["semantic_similar_to"])

    def test_causal_query_weights_caused_by_high(self) -> None:
        weights = self.mem.intent_edge_weights("Why did the permit fail?")
        self.assertGreaterEqual(weights["caused_by"], 0.8)

    def test_available_relation_views_contains_five(self) -> None:
        views = set(self.mem.available_relation_views())
        self.assertEqual(views, {"semantic", "temporal", "causal", "entity", "procedural"})

    def test_uses_single_connection_table_is_true(self) -> None:
        self.assertTrue(self.mem.uses_single_connection_table())

    def test_intent_classifier_routing(self) -> None:
        cases = {
            "Who is the contact?":         "entity",
            "When did this happen?":       "temporal",
            "Why did this fail?":          "causal",
            "How do we estimate panels?":  "procedural",
            "What is the GFCI rule?":      "factual",
        }
        for query, expected in cases.items():
            self.assertEqual(NeuralMemory._classify_intent(query), expected,
                             f"classify_intent({query!r}) != {expected!r}")

    def test_graph_search_empty_db_returns_empty(self) -> None:
        results = self.mem.graph_search("anything", k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
