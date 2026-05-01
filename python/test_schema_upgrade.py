"""Acceptance tests for schema_upgrade.SchemaUpgrade.

Per Phase 7 Commit 1 spec (reference_neural_memory_execution_addendum.md
lines 104-150). Four contracts:

  1. adds 16 memory columns to legacy 8-col schema
  2. adds 10 connection columns to legacy 11-col schema (3 already exist)
  3. is idempotent (running twice produces no change)
  4. preserves existing records (id + content + embedding survive)

Stdlib unittest — repo has no pytest dep. Run:
    python3 -m unittest python.test_schema_upgrade
or:
    python3 python/test_schema_upgrade.py
"""

from __future__ import annotations

import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema_upgrade import (  # noqa: E402
    SchemaUpgrade,
    _CONNECTION_COLUMNS,
    _MEMORY_COLUMNS,
)


# Mirrors the legacy SCHEMA constant in memory_client.py:68-118 — the shape
# the 231-row live DB and any pre-Phase-7 install script produces.
_LEGACY_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT (unixepoch()),
    last_accessed REAL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0
);

CREATE TABLE connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    edge_type TEXT DEFAULT 'similar',
    created_at REAL DEFAULT (unixepoch()),
    event_time REAL DEFAULT NULL,
    ingestion_time REAL DEFAULT NULL,
    valid_from REAL DEFAULT NULL,
    valid_to REAL DEFAULT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);
"""


def _create_legacy_schema(db: Path) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(_LEGACY_SCHEMA)


def _seed_legacy_records(db: Path, count: int = 25) -> list[int]:
    ids: list[int] = []
    with sqlite3.connect(str(db)) as conn:
        for i in range(count):
            blob = struct.pack(f"{4}f", float(i), float(i + 1), float(i + 2), float(i + 3))
            cur = conn.execute(
                "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                (f"label-{i}", f"content body for record {i}", blob),
            )
            ids.append(cur.lastrowid)
        for i in range(min(count - 1, 10)):
            conn.execute(
                "INSERT INTO connections (source_id, target_id, weight) VALUES (?, ?, ?)",
                (ids[i], ids[i + 1], 0.5 + 0.01 * i),
            )
        conn.commit()
    return ids


def _columns(db: Path, table: str) -> set[str]:
    with sqlite3.connect(str(db)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _dump_schema(db: Path) -> str:
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return "\n".join(f"{t}|{n}|{s}" for t, n, s in rows)


class SchemaUpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db = self.tmp_path / "memory.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ----- Contract 1: adds memory columns -------------------------------------
    def test_schema_upgrade_adds_memory_columns(self) -> None:
        _create_legacy_schema(self.db)

        SchemaUpgrade(str(self.db)).upgrade()

        cols = _columns(self.db, "memories")
        expected = {name for name, _ in _MEMORY_COLUMNS}
        missing = expected - cols
        self.assertFalse(missing, f"missing memory cols: {missing}")

        for required in (
            "kind", "confidence", "valid_from", "valid_to", "transaction_time",
            "origin_system", "metadata_json", "memory_visibility", "pin_state",
        ):
            self.assertIn(required, cols)

    # ----- Contract 2: adds connection columns ---------------------------------
    def test_schema_upgrade_adds_connection_columns(self) -> None:
        _create_legacy_schema(self.db)

        SchemaUpgrade(str(self.db)).upgrade()

        cols = _columns(self.db, "connections")
        expected = {name for name, _ in _CONNECTION_COLUMNS}
        missing = expected - cols
        self.assertFalse(missing, f"missing connection cols: {missing}")

        for required in (
            "edge_type", "confidence", "valid_from", "valid_to",
            "transaction_time", "salience", "evidence_count",
        ):
            self.assertIn(required, cols)

    # ----- Contract 3: idempotent ---------------------------------------------
    def test_schema_upgrade_is_idempotent(self) -> None:
        _create_legacy_schema(self.db)
        upgrader = SchemaUpgrade(str(self.db))

        first = upgrader.upgrade()
        schema_after_first = _dump_schema(self.db)

        second = upgrader.upgrade()
        schema_after_second = _dump_schema(self.db)

        self.assertEqual(
            schema_after_first, schema_after_second,
            "schema diverged after second upgrade — not idempotent",
        )
        self.assertEqual(
            second, {"memories_columns_added": 0, "connections_columns_added": 0}
        )
        self.assertEqual(first["memories_columns_added"], 16)
        self.assertEqual(first["connections_columns_added"], 7)

    # ----- Contract 4: preserves existing records -----------------------------
    def test_schema_upgrade_preserves_existing_records(self) -> None:
        _create_legacy_schema(self.db)
        seeded_ids = _seed_legacy_records(self.db, count=25)

        with sqlite3.connect(str(self.db)) as conn:
            before_rows = conn.execute(
                "SELECT id, label, content, embedding, salience, access_count "
                "FROM memories ORDER BY id"
            ).fetchall()
            before_conn = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections ORDER BY id"
            ).fetchall()

        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            after_rows = conn.execute(
                "SELECT id, label, content, embedding, salience, access_count "
                "FROM memories ORDER BY id"
            ).fetchall()
            after_conn = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections ORDER BY id"
            ).fetchall()
            after_ids = [r[0] for r in conn.execute("SELECT id FROM memories ORDER BY id")]

        self.assertEqual(after_ids, seeded_ids, "memory ids changed after migration")
        self.assertEqual(before_rows, after_rows, "memory legacy data changed")
        self.assertEqual(before_conn, after_conn, "connection legacy data changed")

    # ----- Bonus: legacy column shape survives --------------------------------
    def test_legacy_columns_unchanged(self) -> None:
        _create_legacy_schema(self.db)
        legacy_mem = _columns(self.db, "memories")
        legacy_conn = _columns(self.db, "connections")

        SchemaUpgrade(str(self.db)).upgrade()

        after_mem = _columns(self.db, "memories")
        after_conn = _columns(self.db, "connections")

        self.assertTrue(legacy_mem.issubset(after_mem),
                        f"legacy mem cols lost: {legacy_mem - after_mem}")
        self.assertTrue(legacy_conn.issubset(after_conn),
                        f"legacy connection cols lost: {legacy_conn - after_conn}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
