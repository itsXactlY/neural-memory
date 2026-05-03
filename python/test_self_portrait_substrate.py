"""Tests for self_portrait_substrate.py — agent self-portrait substrate read helpers.

Per packet S-PORTRAIT-1 acceptance criteria:
- agent attribution filter actually applied
- kind='self_portrait' / kind='reflection' tolerated when absent from schema
- top_entities uses connections graph (joined)
- dream_insights returns kind='dream_insight' (or 'insight') rows
- peer_portraits excludes self
- compose_substrate_packet returns the full 6-key structure
- every helper handles empty substrate without crashing
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402
import self_portrait_substrate as sps  # noqa: E402


# ---------------------------------------------------------------------------
# MagicMock substrate fixture — for tests that want to inspect the SQL/params
# the helpers emit, without booting a real NeuralMemory.
# ---------------------------------------------------------------------------


class _MockStore:
    """Stand-in for SQLiteStore that records every query."""

    def __init__(self, rows_by_query: list[list[tuple]] | None = None):
        self.queries: list[tuple[str, tuple]] = []
        self._rows_by_query = list(rows_by_query or [])
        self._lock = MagicMock()
        self._lock.__enter__ = MagicMock(return_value=None)
        self._lock.__exit__ = MagicMock(return_value=None)
        self.conn = MagicMock()
        self.conn.execute = self._execute

    def _execute(self, sql: str, params: tuple = ()):
        self.queries.append((sql, params))
        cursor = MagicMock()
        if self._rows_by_query:
            rows = self._rows_by_query.pop(0)
        else:
            rows = []
        cursor.fetchall.return_value = rows
        return cursor


class _MockMem:
    def __init__(self, rows_by_query: list[list[tuple]] | None = None):
        self.store = _MockStore(rows_by_query)


def _row(
    *,
    id_: int = 1,
    label: str = "",
    content: str = "",
    kind: str = "experience",
    salience: float = 1.0,
    created_at: float = 1000.0,
    last_accessed: float = 1000.0,
    origin_system: str | None = None,
    source: str | None = None,
    metadata: dict | None = None,
) -> tuple:
    """Build a 10-tuple matching _BASE_COLS column order."""
    md = json.dumps(metadata) if metadata else None
    return (
        id_, label, content, kind, salience, created_at,
        last_accessed, origin_system, source, md,
    )


# ---------------------------------------------------------------------------
# In-memory NeuralMemory fixture — for end-to-end "empty substrate" smoke.
# ---------------------------------------------------------------------------


class _LiveMemFixture:
    """Boots a real NeuralMemory on a tempdir DB. Used for integration-flavored
    contracts (empty-substrate graceful, peer-portrait isolation across kinds)."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )

    def close(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ReadSelfRelevantMemoriesTests(unittest.TestCase):
    def test_sql_filter_includes_agent_aliases_and_kind_exclusion(self) -> None:
        # Filter is at SQL-level now (not post-filter) — verify the WHERE
        # clause covers both the kind exclusion AND the multi-surface agent
        # alias predicate (origin_system / source / metadata LIKE patterns).
        mem = _MockMem([[]])
        sps.read_self_relevant_memories(mem, "valiendo", limit=10)
        sql, params = mem.store.queries[0]
        self.assertIn("kind NOT IN ('entity', 'self_portrait', 'reflection')", sql)
        self.assertIn("LOWER(origin_system) = ?", sql)
        self.assertIn("LOWER(source) = ?", sql)
        self.assertIn("metadata_json LIKE ?", sql)
        # Every alias of 'valiendo' must appear in the params list.
        for alias in ("valiendo", "valiendo-hermes", "valiendo_hermes", "hermes"):
            self.assertIn(alias, params,
                          f"alias {alias!r} missing from SQL params")
        # And each metadata-key pattern is present for at least one alias.
        for key in ("from", "author", "actor", "agent", "agent_name"):
            self.assertTrue(
                any(isinstance(p, str) and f'"{key}":' in p for p in params),
                f"no LIKE pattern for metadata.{key}",
            )

    def test_returns_rows_db_provides(self) -> None:
        # With SQL filtering, helper trusts what DB returns. Verify rows
        # are dict-shaped per spec (id, content, ts, kind, salience).
        rows = [
            _row(id_=1, kind="experience", origin_system="hermes",
                 source="bridge_mailbox", content="hermes msg"),
            _row(id_=2, kind="experience",
                 metadata={"from": "valiendo-hermes"}, content="from valiendo"),
        ]
        mem = _MockMem([rows])
        out = sps.read_self_relevant_memories(mem, "valiendo", limit=10)
        self.assertEqual(len(out), 2)
        for d in out:
            for k in ("id", "content", "ts", "kind", "salience"):
                self.assertIn(k, d)

    def test_zero_limit_returns_empty_without_query(self) -> None:
        mem = _MockMem([[_row(id_=1, origin_system="hermes")]])
        self.assertEqual(sps.read_self_relevant_memories(mem, "valiendo", limit=0), [])
        # Must short-circuit without running any query.
        self.assertEqual(len(mem.store.queries), 0)

    def test_unknown_agent_name_passes_through_as_token(self) -> None:
        # Arbitrary agent name (no allowlist, packet rule 6). The token
        # itself is what the SQL filter pivots on.
        mem = _MockMem([[]])
        sps.read_self_relevant_memories(mem, "custom_agent", limit=5)
        _, params = mem.store.queries[0]
        self.assertIn("custom_agent", params)


class ReadRecentReflectionsTests(unittest.TestCase):
    def test_handles_missing_kind(self) -> None:
        # No rows returned because kind='self_portrait' / 'reflection' don't
        # exist in the schema yet. Helper must return [] not crash.
        mem = _MockMem([[]])
        out = sps.read_recent_reflections(mem, "valiendo", limit=5)
        self.assertEqual(out, [])
        sql, _ = mem.store.queries[0]
        self.assertIn("kind IN ('reflection', 'self_portrait')", sql)

    def test_sql_filter_includes_agent_aliases(self) -> None:
        # SQL-level filtering — verify the WHERE has both kind IN and
        # alias predicate.
        mem = _MockMem([[]])
        sps.read_recent_reflections(mem, "valiendo", limit=5)
        sql, params = mem.store.queries[0]
        self.assertIn("kind IN ('reflection', 'self_portrait')", sql)
        self.assertIn("LOWER(origin_system) = ?", sql)
        for alias in ("valiendo", "hermes"):
            self.assertIn(alias, params)


class ReadTopEntitiesTests(unittest.TestCase):
    def test_uses_connections_graph(self) -> None:
        # First query: agent's recent memory ids (now SQL-filtered).
        recent = [
            _row(id_=100, origin_system="hermes", kind="experience"),
            _row(id_=101, origin_system="hermes", kind="experience"),
        ]
        # Second query: the entity-join. Returns 11-tuple (10 cols + sum).
        entity_rows = [
            (200, "lennar", "Entity: Lennar", "entity", 1.0, 999.0, 999.0,
             "entity_extractor", None, None, 4.5),
            (201, "tito", "Entity: Tito", "entity", 1.0, 998.0, 998.0,
             "entity_extractor", None, None, 3.2),
        ]
        mem = _MockMem([recent, entity_rows])
        out = sps.read_top_entities(mem, "valiendo", limit=5)
        self.assertEqual([m["id"] for m in out], [200, 201])
        self.assertEqual(out[0]["edge_weight_sum"], 4.5)

        # Verify the join actually queried the connections table.
        join_sql, _ = mem.store.queries[1]
        self.assertIn("FROM connections", join_sql)
        self.assertIn("JOIN memories", join_sql)
        self.assertIn("kind = 'entity'", join_sql)

        # And the first (recent-memories) query carries the agent alias predicate.
        recent_sql, recent_params = mem.store.queries[0]
        self.assertIn("kind != 'entity'", recent_sql)
        self.assertIn("LOWER(origin_system) = ?", recent_sql)
        self.assertIn("hermes", recent_params)

    def test_no_recent_memories_returns_empty_without_join_query(self) -> None:
        # Recent query returns nothing for this agent → skip the join entirely.
        mem = _MockMem([[]])
        out = sps.read_top_entities(mem, "valiendo", limit=5)
        self.assertEqual(out, [])
        # Only the recent-memory query should have run.
        self.assertEqual(len(mem.store.queries), 1)


class ReadRecentDreamInsightsTests(unittest.TestCase):
    def test_returns_kind_dream_insight(self) -> None:
        rows = [
            _row(id_=300, kind="dream_insight", content="cluster insight A"),
            _row(id_=301, kind="dream_insight", content="cluster insight B"),
        ]
        mem = _MockMem([rows])
        out = sps.read_recent_dream_insights(mem, limit=5)
        self.assertEqual([m["id"] for m in out], [300, 301])
        sql, _ = mem.store.queries[0]
        # Accepts both the live name ('dream_insight') and the spec name ('insight').
        self.assertIn("dream_insight", sql)
        self.assertIn("insight", sql)

    def test_zero_limit_returns_empty(self) -> None:
        mem = _MockMem([])
        self.assertEqual(sps.read_recent_dream_insights(mem, limit=0), [])


class ReadPeerPortraitsTests(unittest.TestCase):
    def test_excludes_self_agent(self) -> None:
        # Mix of portraits authored by various agents. 'valiendo' (and aliases)
        # MUST NOT appear in the returned dict.
        rows = [
            _row(id_=400, kind="self_portrait", origin_system="hermes",
                 content="my own portrait"),
            _row(id_=401, kind="self_portrait", origin_system="claude_memory",
                 content="claude portrait"),
            _row(id_=402, kind="self_portrait",
                 metadata={"from": "codex"}, content="codex portrait"),
            _row(id_=403, kind="self_portrait",
                 metadata={"author": "valiendo"}, content="alias self"),
        ]
        mem = _MockMem([rows])
        out = sps.read_peer_portraits(mem, exclude_agent="valiendo", limit=3)
        self.assertNotIn("valiendo", out)
        # Hermes alias of valiendo must also not appear.
        self.assertNotIn("hermes", out)
        self.assertIn("claude-code", out)
        self.assertIn("codex", out)
        # The metadata.author='valiendo' row must not slip through under any key.
        for portraits in out.values():
            for p in portraits:
                self.assertNotEqual(p["id"], 400)
                self.assertNotEqual(p["id"], 403)

    def test_empty_substrate_returns_empty_dict(self) -> None:
        mem = _MockMem([[]])
        self.assertEqual(sps.read_peer_portraits(mem, "valiendo"), {})


class ComposeSubstratePacketTests(unittest.TestCase):
    def test_returns_complete_structure(self) -> None:
        # All queries return [] — we only care about packet shape here.
        mem = _MockMem([[]] * 10)
        packet = sps.compose_substrate_packet(mem, "valiendo")
        self.assertEqual(
            set(packet.keys()),
            {
                "agent",
                "ts",
                "self_memories",
                "self_reflections",
                "top_entities",
                "dream_insights",
                "peer_portraits",
            },
        )
        self.assertEqual(packet["agent"], "valiendo")
        self.assertIsInstance(packet["ts"], float)
        self.assertEqual(packet["self_memories"], [])
        self.assertEqual(packet["self_reflections"], [])
        self.assertEqual(packet["top_entities"], [])
        self.assertEqual(packet["dream_insights"], [])
        self.assertEqual(packet["peer_portraits"], {})


class EmptySubstrateGracefulTests(unittest.TestCase):
    """End-to-end smoke against a real (but empty) NeuralMemory. No data
    means every helper must return empty container, never None / never raise."""

    def setUp(self) -> None:
        self.fixture = _LiveMemFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_all_helpers_handle_empty_substrate_gracefully(self) -> None:
        mem = self.fixture.mem
        for agent in ("claude-code", "valiendo", "codex", "novel-agent-name"):
            self.assertEqual(sps.read_self_relevant_memories(mem, agent), [])
            self.assertEqual(sps.read_recent_reflections(mem, agent), [])
            self.assertEqual(sps.read_top_entities(mem, agent), [])
            self.assertEqual(sps.read_peer_portraits(mem, agent), {})
        self.assertEqual(sps.read_recent_dream_insights(mem), [])
        packet = sps.compose_substrate_packet(mem, "claude-code")
        self.assertEqual(packet["self_memories"], [])
        self.assertEqual(packet["self_reflections"], [])
        self.assertEqual(packet["top_entities"], [])
        self.assertEqual(packet["dream_insights"], [])
        self.assertEqual(packet["peer_portraits"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
