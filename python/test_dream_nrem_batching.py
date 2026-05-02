"""Regression coverage for D6 NREM SQLite batching.

The first real 03:45 launchd run proved a performance bug rather than a launchd
bug: NREM iterated the full active graph while SQLiteDreamBackend opened a fresh
connection and commit for each edge update + history row. This test locks the
fix to O(1) backend connections per NREM pass instead of O(N) with edge count.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream_engine import DreamEngine  # noqa: E402
from memory_client import NeuralMemory  # noqa: E402


class DreamNremBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )
        self.engine = DreamEngine.sqlite(
            self.db_path,
            neural_memory=self.mem,
            max_memories_per_cycle=20,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_phase_nrem_uses_constant_backend_connections_for_edge_updates(self) -> None:
        memory_ids = [
            self.mem.remember(
                f"Valiendo D6 batching seed memory {idx}",
                detect_conflicts=False,
                kind="experience",
            )
            for idx in range(6)
        ]
        for idx, source_id in enumerate(memory_ids):
            for target_id in memory_ids[idx + 1 :]:
                self.mem.store.add_connection(source_id, target_id, 0.8, edge_type="similar")

        backend = self.engine._backend
        real_connect = backend._connect
        connect_calls = 0

        def counted_connect():
            nonlocal connect_calls
            connect_calls += 1
            return real_connect()

        backend._connect = counted_connect
        try:
            stats = self.engine._phase_nrem()
        finally:
            backend._connect = real_connect

        self.assertGreater(
            stats["strengthened"] + stats["weakened"],
            0,
            f"expected NREM to update at least one edge, got {stats}",
        )
        self.assertLessEqual(
            connect_calls,
            6,
            f"expected O(1) backend connections for NREM batching, got {connect_calls}",
        )
        with self.mem.store._lock:
            history_count = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM connection_history"
            ).fetchone()[0]
        self.assertGreater(
            history_count,
            0,
            "batched NREM updates must still write connection history rows",
        )


    def test_phase_nrem_flushes_update_batches_before_full_graph_buffering(self) -> None:
        """NM_NREM_UPDATE_BATCH bounds the NREM updates buffer.

        Per per-commit reviewer 8b26966 + resolver 070621: NREM was still
        accumulating one tuple per edge before flush (770MB+ on 7.7M edges).
        Now flushes in chunks of NM_NREM_UPDATE_BATCH (default 50K).
        """
        memory_ids = [
            self.mem.remember(
                f"D6 bounded update seed memory {idx}",
                detect_conflicts=False,
                kind="experience",
            )
            for idx in range(7)
        ]
        for idx, source_id in enumerate(memory_ids):
            for target_id in memory_ids[idx + 1 :]:
                self.mem.store.add_connection(source_id, target_id, 0.8, edge_type="similar")

        backend = self.engine._backend
        real_apply = backend.apply_connection_changes
        batch_lengths = []

        def counted_apply(updates, *, prune_threshold=0.05):
            batch_lengths.append(len(updates))
            return real_apply(updates, prune_threshold=prune_threshold)

        prior = os.environ.get("NM_NREM_UPDATE_BATCH")
        os.environ["NM_NREM_UPDATE_BATCH"] = "4"
        backend.apply_connection_changes = counted_apply
        try:
            stats = self.engine._phase_nrem()
        finally:
            backend.apply_connection_changes = real_apply
            if prior is None:
                os.environ.pop("NM_NREM_UPDATE_BATCH", None)
            else:
                os.environ["NM_NREM_UPDATE_BATCH"] = prior

        self.assertGreater(
            stats["strengthened"] + stats["weakened"], 0,
            f"expected NREM to update at least one edge, got {stats}",
        )
        non_empty_batches = [n for n in batch_lengths if n > 0]
        self.assertGreater(
            len(non_empty_batches), 1,
            f"expected multiple bounded update flushes, got {batch_lengths}",
        )
        self.assertLessEqual(
            max(non_empty_batches), 4,
            f"NREM buffered more than NM_NREM_UPDATE_BATCH updates: {batch_lengths}",
        )

    def test_iter_connections_works_on_legacy_schema_without_valid_to(self) -> None:
        """Pre-bi-temporal DBs lack valid_to; iter_connections must still work.
        Caught by per-commit reviewer 8b26966 — original SQL hard-required the
        column, would crash on legacy substrates with `no such column: valid_to`.
        """
        # Drop the valid_to column to simulate legacy schema. SQLite < 3.35
        # doesn't support DROP COLUMN, so use a CREATE...AS SELECT trick.
        with self.mem.store._lock:
            self.mem.store.conn.executescript("""
                CREATE TABLE connections_legacy AS
                    SELECT id, source_id, target_id, weight, edge_type, created_at
                    FROM connections;
                DROP TABLE connections;
                ALTER TABLE connections_legacy RENAME TO connections;
            """)
            self.mem.store.conn.execute(
                "INSERT INTO connections (source_id, target_id, weight) VALUES (1, 2, 0.8)"
            )
            self.mem.store.conn.commit()

        backend = self.engine._backend
        rows = list(backend.iter_connections())
        self.assertGreater(
            len(rows), 0,
            "iter_connections must yield rows on legacy schema (no valid_to)",
        )

    def test_phase_nrem_skips_connection_history_when_disabled(self) -> None:
        """NM_DISABLE_CONN_HISTORY=1 must short-circuit history INSERTs.

        Sonnet investigation 2026-05-02 [verified-now]: zero production
        code reads connection_history. The gate prevents 7.7M+ INSERTs
        per cycle from accruing dead audit weight (61.5M pre-fix).
        """
        import os
        memory_ids = [
            self.mem.remember(
                f"D6 history-gate seed memory {idx}",
                detect_conflicts=False,
                kind="experience",
            )
            for idx in range(4)
        ]
        for idx, source_id in enumerate(memory_ids):
            for target_id in memory_ids[idx + 1 :]:
                self.mem.store.add_connection(source_id, target_id, 0.8, edge_type="similar")

        prior = os.environ.get("NM_DISABLE_CONN_HISTORY")
        os.environ["NM_DISABLE_CONN_HISTORY"] = "1"
        try:
            stats = self.engine._phase_nrem()
        finally:
            if prior is None:
                os.environ.pop("NM_DISABLE_CONN_HISTORY", None)
            else:
                os.environ["NM_DISABLE_CONN_HISTORY"] = prior

        self.assertGreater(
            stats["strengthened"] + stats["weakened"], 0,
            "NREM should still process edges with history gate on",
        )
        with self.mem.store._lock:
            history_count = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM connection_history"
            ).fetchone()[0]
        self.assertEqual(
            history_count, 0,
            "NM_DISABLE_CONN_HISTORY=1 must skip all connection_history writes",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
