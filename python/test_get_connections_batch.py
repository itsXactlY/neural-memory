"""Tests for get_connections_batch — batched edge fetching.

Per reviewer-round-7 LOW finding: get_connections_batch is new
critical-path code (BFS frontier expansion in graph_search) with
zero coverage. These contracts cover:
- Single-node returns same edges as get_connections
- Multi-node returns dict keyed by node_id
- Self-loop dedup (r[1] != r[0] guard)
- Bidirectional distribution (edge appears under both endpoints)
- Empty input returns empty dict
- Expired edge filtering matches get_connections behavior
- include_expired=True returns all
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class GetConnectionsBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )
        self.a = self.mem.remember("memory A", detect_conflicts=False)
        self.b = self.mem.remember("memory B", detect_conflicts=False)
        self.c = self.mem.remember("memory C", detect_conflicts=False)
        self.d = self.mem.remember("memory D (isolated)", detect_conflicts=False)

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        os.unlink(self.db_path)

    def _add_edge(self, src: int, tgt: int, weight: float = 0.8,
                   edge_type: str = "test"):
        self.mem.store.add_connection(src, tgt, weight=weight,
                                       edge_type=edge_type)

    def test_empty_input_returns_empty_dict(self) -> None:
        result = self.mem.store.get_connections_batch([])
        self.assertEqual(result, {})

    def test_single_node_matches_get_connections(self) -> None:
        self._add_edge(self.a, self.b, weight=0.9, edge_type="t1")
        self._add_edge(self.a, self.c, weight=0.7, edge_type="t2")
        single = self.mem.store.get_connections(self.a)
        batch = self.mem.store.get_connections_batch([self.a])
        self.assertIn(self.a, batch)
        self.assertEqual(
            len(single), len(batch[self.a]),
            "single-node batch must return same edges as get_connections",
        )

    def test_multi_node_returns_dict_keyed_by_id(self) -> None:
        self._add_edge(self.a, self.b)
        self._add_edge(self.c, self.d)
        result = self.mem.store.get_connections_batch([self.a, self.c, self.d])
        self.assertEqual(set(result.keys()), {self.a, self.c, self.d})
        self.assertEqual(len(result[self.a]), 1)
        self.assertEqual(len(result[self.c]), 1)
        self.assertEqual(len(result[self.d]), 1)

    def test_bidirectional_distribution(self) -> None:
        # Edge a→b: when we batch [a, b], edge appears under BOTH endpoints
        self._add_edge(self.a, self.b)
        result = self.mem.store.get_connections_batch([self.a, self.b])
        self.assertEqual(len(result[self.a]), 1)
        self.assertEqual(len(result[self.b]), 1)
        # Same edge dict semantically
        self.assertEqual(result[self.a][0]['source'], self.a)
        self.assertEqual(result[self.b][0]['source'], self.a)

    def test_self_loop_not_double_counted(self) -> None:
        # Edge a→a should appear ONCE in result[a], not twice
        self._add_edge(self.a, self.a, weight=0.5, edge_type="self")
        result = self.mem.store.get_connections_batch([self.a])
        # Filter to self-edges
        self_edges = [e for e in result[self.a]
                      if e['source'] == self.a and e['target'] == self.a]
        self.assertEqual(len(self_edges), 1,
                         "self-loop must not be double-counted via the "
                         "r[1] != r[0] guard")

    def test_isolated_node_returns_empty_list(self) -> None:
        # Node d has no edges; should appear in dict with []
        result = self.mem.store.get_connections_batch([self.d])
        self.assertEqual(result[self.d], [])

    def test_node_not_in_request_excluded(self) -> None:
        # If we batch [a] but the edge is c→b, neither side appears in
        # result (only requested nodes are in output dict)
        self._add_edge(self.c, self.b)
        result = self.mem.store.get_connections_batch([self.a])
        self.assertEqual(result, {self.a: []})

    def test_expired_edge_filtered_by_default(self) -> None:
        # Add edge then mark it expired
        self._add_edge(self.a, self.b, weight=0.9, edge_type="expiring")
        # Mark all a's edges expired now
        self.mem.store.set_edges_valid_to(self.a, time.time() - 1)
        result = self.mem.store.get_connections_batch([self.a])
        # Should be filtered out
        expiring = [e for e in result[self.a] if e['type'] == 'expiring']
        self.assertEqual(len(expiring), 0,
                         "expired edges should be filtered by default")

    def test_include_expired_returns_all(self) -> None:
        self._add_edge(self.a, self.b, weight=0.9, edge_type="expiring")
        self.mem.store.set_edges_valid_to(self.a, time.time() - 1)
        result = self.mem.store.get_connections_batch(
            [self.a], include_expired=True,
        )
        # Should include the expired edge
        expiring = [e for e in result[self.a] if e['type'] == 'expiring']
        self.assertEqual(len(expiring), 1,
                         "include_expired=True must return expired edges")


if __name__ == "__main__":
    unittest.main()
