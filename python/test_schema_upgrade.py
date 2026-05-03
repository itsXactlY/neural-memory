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
    _EVIDENCE_LEDGER_TARGET_USER_VERSION,
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
        # Second-run column adds must be zero (idempotency proof). Other keys
        # may be present (e.g., fts_rows_backfilled added in P7C5) — only the
        # column-add counts are part of the idempotency contract.
        self.assertEqual(second["memories_columns_added"], 0)
        self.assertEqual(second["connections_columns_added"], 0)
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

    # ----- Defensive FTS cleanup (Phase 7 fix c2c2321) -----------------------
    def test_ensure_fts5_cleans_entity_rows_defensively(self) -> None:
        """Phase 7 fix c2c2321: SchemaUpgrade._ensure_fts5() must DELETE any
        kind='entity' rows that snuck into memories_fts via a long-running
        process holding stale memory_client.py in memory.

        Simulate stale-code path: insert entity row + manually pollute FTS5.
        Re-run SchemaUpgrade — defensive cleanup must remove the entity row."""
        _create_legacy_schema(self.db)
        SchemaUpgrade(str(self.db)).upgrade()  # first run: schema + FTS5 ready

        with sqlite3.connect(str(self.db)) as conn:
            cur = conn.execute(
                "INSERT INTO memories (label, content, kind, salience) "
                "VALUES (?, ?, ?, ?)",
                ("Lennar", "Entity: Lennar", "entity", 1.0),
            )
            entity_id = cur.lastrowid
            # Manually pollute FTS5 (simulating stale code path)
            conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (entity_id, "Entity: Lennar"),
            )
            conn.commit()
            count_before = conn.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE rowid = ?", (entity_id,)
            ).fetchone()[0]
            self.assertEqual(count_before, 1)

        # Re-run SchemaUpgrade — defensive cleanup fires
        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE rowid = ?", (entity_id,)
            ).fetchone()[0]
            self.assertEqual(count_after, 0,
                             "defensive _ensure_fts5 cleanup did not remove "
                             "stale entity row from memories_fts")

    def test_ensure_fts5_backfill_skips_entity_rows(self) -> None:
        """Backfill must NOT add kind='entity' rows to memories_fts."""
        _create_legacy_schema(self.db)
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute(
                "INSERT INTO memories (label, content, salience) VALUES (?, ?, ?)",
                ("user-mem", "regular content", 1.0),
            )
            conn.execute("ALTER TABLE memories ADD COLUMN kind TEXT")
            conn.execute(
                "INSERT INTO memories (label, content, kind, salience) "
                "VALUES (?, ?, ?, ?)",
                ("Lennar", "Entity: Lennar", "entity", 1.0),
            )
            conn.commit()

        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            entity_in_fts = conn.execute(
                "SELECT COUNT(*) FROM memories_fts m "
                "JOIN memories mem ON mem.id = m.rowid "
                "WHERE mem.kind = 'entity'"
            ).fetchone()[0]
            user_in_fts = conn.execute(
                "SELECT COUNT(*) FROM memories_fts m "
                "JOIN memories mem ON mem.id = m.rowid "
                "WHERE mem.kind IS NULL OR mem.kind != 'entity'"
            ).fetchone()[0]
            self.assertEqual(entity_in_fts, 0,
                             "kind-aware backfill must skip entity rows")
            self.assertEqual(user_in_fts, 1,
                             "user memory should be backfilled into FTS")


    # ----- Evidence ledger (S2 packet 2026-05-03) -----------------------------
    def test_evidence_ledger_idempotent_user_version_bump(self) -> None:
        """First upgrade bumps user_version 0 -> target. Second upgrade is
        a no-op (user_version stays at target, no error)."""
        _create_legacy_schema(self.db)
        with sqlite3.connect(str(self.db)) as conn:
            uv_before = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(uv_before, 0,
                         "legacy DB must start at user_version 0")

        first = SchemaUpgrade(str(self.db)).upgrade()
        self.assertEqual(first["user_version_before"], 0)
        self.assertEqual(first["user_version_after"],
                         _EVIDENCE_LEDGER_TARGET_USER_VERSION)
        self.assertEqual(first["ledger_indexes_created"], 2)

        with sqlite3.connect(str(self.db)) as conn:
            uv_mid = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(uv_mid, _EVIDENCE_LEDGER_TARGET_USER_VERSION)

        second = SchemaUpgrade(str(self.db)).upgrade()
        self.assertEqual(second["user_version_before"],
                         _EVIDENCE_LEDGER_TARGET_USER_VERSION)
        self.assertEqual(second["user_version_after"],
                         _EVIDENCE_LEDGER_TARGET_USER_VERSION)
        self.assertEqual(second["ledger_indexes_created"], 0,
                         "second run must be a no-op (zero indexes created)")

        with sqlite3.connect(str(self.db)) as conn:
            uv_after = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(uv_after, _EVIDENCE_LEDGER_TARGET_USER_VERSION)

    def test_evidence_ledger_table_columns_and_indexes(self) -> None:
        """Ledger table has all required columns + 2 UNIQUE indexes after
        upgrade. Schema must match the v2 contract (S2b 2026-05-03)."""
        _create_legacy_schema(self.db)
        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            cols = {row[1]: row for row in
                    conn.execute("PRAGMA table_info(evidence_ledger)")}
            indexes = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='index' AND tbl_name='evidence_ledger' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

        expected_cols = {
            "evidence_id", "memory_id", "evidence_type",
            "source_system", "source_record_id", "status",
            "inserted_at", "updated_at", "metadata_hash",
        }
        self.assertEqual(set(cols.keys()), expected_cols,
                         f"ledger column set mismatch: got {set(cols.keys())}")

        # PK + NOT NULL constraints on key contract columns
        # cols[name] = (cid, name, type, notnull, dflt, pk)
        self.assertEqual(cols["evidence_id"][5], 1, "evidence_id must be PK")
        self.assertEqual(cols["evidence_type"][3], 1, "evidence_type NOT NULL")
        self.assertEqual(cols["source_system"][3], 1, "source_system NOT NULL")
        self.assertEqual(cols["source_record_id"][3], 1,
                         "source_record_id NOT NULL")
        self.assertEqual(cols["status"][3], 1, "status NOT NULL")

        # 2 UNIQUE indexes per v2 contract (S2b 2026-05-03):
        #   • idx_evidence_ledger_source              — (system, record_id)
        #   • idx_evidence_ledger_type_source_record  — (type, system, record_id)
        # The legacy v1 index (idx_evidence_ledger_type_record) was migrated
        # away to drop the global-uniqueness assumption — see S2b packet.
        index_names = {row[0] for row in indexes}
        self.assertIn("idx_evidence_ledger_source", index_names)
        self.assertIn("idx_evidence_ledger_type_source_record", index_names)
        self.assertNotIn("idx_evidence_ledger_type_record", index_names,
                         "legacy v1 index must be absent after v2 migration")
        for name, sql in indexes:
            if name in ("idx_evidence_ledger_source",
                        "idx_evidence_ledger_type_source_record"):
                self.assertIn("UNIQUE", sql.upper(),
                              f"index {name} must be UNIQUE: {sql}")

    def test_evidence_ledger_status_check_constraint(self) -> None:
        """status column has CHECK constraint allowing only inserted /
        superseded / retracted."""
        _create_legacy_schema(self.db)
        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            # Valid values insert cleanly. Each row needs a distinct
            # source_record_id to avoid the (evidence_type, source_record_id)
            # unique-index collision.
            for status in ("inserted", "superseded", "retracted"):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, "
                    " source_record_id, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (f"id-{status}", "wa_crew_message", "x",
                     f"rec-{status}", status),
                )
            # Invalid value is rejected by CHECK.
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, "
                    " source_record_id, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("id-bad", "wa_crew_message", "x", "rec-bad", "garbage"),
                )

    def test_evidence_ledger_unique_indexes_enforce(self) -> None:
        """v2 contract (S2b 2026-05-03):
          • idx_evidence_ledger_source: (source_system, source_record_id) UNIQUE
            — blocks the same source from inserting the same record_id twice.
          • idx_evidence_ledger_type_source_record:
            (evidence_type, source_system, source_record_id) UNIQUE
            — blocks the same triple twice.
        Crucially, the v1 assumption that source_record_id is globally unique
        across source_systems is RELAXED. Two different source_systems may
        emit the same source_record_id under the same evidence_type
        (e.g., evidence_type='sent_pdf' from both
        'sent_estimate_pdf_miner' and 'ae_dashboard') — they get distinct
        evidence_ids and distinct rows.
        """
        _create_legacy_schema(self.db)
        SchemaUpgrade(str(self.db)).upgrade()
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute(
                "INSERT INTO evidence_ledger "
                "(evidence_id, evidence_type, source_system, source_record_id) "
                "VALUES (?, ?, ?, ?)",
                ("aaa", "wa_crew_message", "hermes_wa_bridge", "rec-1"),
            )
            # Same (source_system, source_record_id) — blocked by per-source
            # identity index (correct: same source can't claim same record id
            # twice).
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, source_record_id) "
                    "VALUES (?, ?, ?, ?)",
                    ("bbb", "sent_pdf", "hermes_wa_bridge", "rec-1"),
                )
            # Same (evidence_type, source_record_id) but DIFFERENT source_system
            # — ALLOWED in v2. v1 would have blocked this; v2 explicitly
            # allows it because each source_system owns its own namespace.
            conn.execute(
                "INSERT INTO evidence_ledger "
                "(evidence_id, evidence_type, source_system, source_record_id) "
                "VALUES (?, ?, ?, ?)",
                ("ccc", "wa_crew_message", "ae_dashboard", "rec-1"),
            )
            # ...but the same (evidence_type, source_system, source_record_id)
            # triple twice IS blocked by the wider v2 index.
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, source_record_id) "
                    "VALUES (?, ?, ?, ?)",
                    ("ddd", "wa_crew_message", "ae_dashboard", "rec-1"),
                )

    # ----- v1 → v2 migration (S2b 2026-05-03) ---------------------------------
    def _create_v1_ledger(self) -> None:
        """Recreate the v1-shape ledger by hand: legacy index name +
        user_version=1. Used to validate v1→v2 in-place migration on a DB
        that originally received the v1 layout."""
        _create_legacy_schema(self.db)
        with sqlite3.connect(str(self.db)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_ledger (
                    evidence_id      TEXT PRIMARY KEY,
                    memory_id        INTEGER,
                    evidence_type    TEXT NOT NULL,
                    source_system    TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    status           TEXT NOT NULL DEFAULT 'inserted'
                                       CHECK (status IN ('inserted','superseded','retracted')),
                    inserted_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata_hash    TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ledger_source
                    ON evidence_ledger (source_system, source_record_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ledger_type_record
                    ON evidence_ledger (evidence_type, source_record_id);
                PRAGMA user_version = 1;
                """
            )

    def test_v1_to_v2_migration_drops_legacy_creates_replacement(self) -> None:
        """Applying upgrade to an existing v1 ledger: drops
        idx_evidence_ledger_type_record, creates
        idx_evidence_ledger_type_source_record, bumps user_version 1→2."""
        self._create_v1_ledger()

        with sqlite3.connect(str(self.db)) as conn:
            uv_before = conn.execute("PRAGMA user_version").fetchone()[0]
            idx_before = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='index' AND tbl_name='evidence_ledger' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
        self.assertEqual(uv_before, 1)
        self.assertIn("idx_evidence_ledger_type_record", idx_before)
        self.assertNotIn("idx_evidence_ledger_type_source_record", idx_before)

        result = SchemaUpgrade(str(self.db)).upgrade()
        self.assertEqual(result["user_version_before"], 1)
        self.assertEqual(result["user_version_after"],
                         _EVIDENCE_LEDGER_TARGET_USER_VERSION)

        with sqlite3.connect(str(self.db)) as conn:
            uv_after = conn.execute("PRAGMA user_version").fetchone()[0]
            idx_after = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='index' AND tbl_name='evidence_ledger' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
        self.assertEqual(uv_after, 2)
        self.assertNotIn("idx_evidence_ledger_type_record", idx_after,
                         "legacy v1 index must be dropped")
        self.assertIn("idx_evidence_ledger_type_source_record", idx_after,
                      "v2 replacement index must exist")
        self.assertIn("idx_evidence_ledger_source", idx_after,
                      "per-source index must survive migration")

    def test_v1_to_v2_migration_idempotent(self) -> None:
        """Re-running upgrade after v1→v2 migration is a no-op (already at
        target version, no schema churn)."""
        self._create_v1_ledger()
        first = SchemaUpgrade(str(self.db)).upgrade()
        self.assertEqual(first["user_version_after"], 2)
        schema_after_first = _dump_schema(self.db)

        second = SchemaUpgrade(str(self.db)).upgrade()
        self.assertEqual(second["user_version_before"], 2)
        self.assertEqual(second["user_version_after"], 2)
        self.assertEqual(second["ledger_indexes_created"], 0,
                         "post-migration re-run must touch zero indexes")
        schema_after_second = _dump_schema(self.db)
        self.assertEqual(schema_after_first, schema_after_second,
                         "v2 schema diverged on second run — not idempotent")

    def test_v1_to_v2_migration_preserves_pre_existing_ledger_rows(self) -> None:
        """A v1 ledger with some rows must keep them all after v1→v2 migration.
        Wider replacement index has strict-superset semantics — anything
        unique under (evidence_type, source_record_id) is also unique under
        (evidence_type, source_system, source_record_id), so existing rows
        cannot violate the new constraint."""
        self._create_v1_ledger()
        with sqlite3.connect(str(self.db)) as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, source_record_id) "
                    "VALUES (?, ?, ?, ?)",
                    (f"v1-id-{i}", "wa_crew_message", "hermes_wa_bridge",
                     f"v1-rec-{i}"),
                )
            conn.commit()
            before = conn.execute(
                "SELECT evidence_id, evidence_type, source_system, "
                "source_record_id FROM evidence_ledger ORDER BY evidence_id"
            ).fetchall()

        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            after = conn.execute(
                "SELECT evidence_id, evidence_type, source_system, "
                "source_record_id FROM evidence_ledger ORDER BY evidence_id"
            ).fetchall()
        self.assertEqual(before, after,
                         "v1→v2 migration lost or mutated pre-existing rows")
        self.assertEqual(len(after), 5)

    def test_v1_to_v2_migration_unblocks_multi_source_collision(self) -> None:
        """The whole point of v1→v2: after migration, two rows with same
        (evidence_type, source_record_id) but DIFFERENT source_system
        coexist. Pre-migration this would have raised IntegrityError; post-
        migration it succeeds."""
        self._create_v1_ledger()
        # In v1, this pair would collide. Insert only the first row so the
        # ledger is non-empty when we migrate; the colliding partner goes in
        # AFTER the upgrade.
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute(
                "INSERT INTO evidence_ledger "
                "(evidence_id, evidence_type, source_system, source_record_id) "
                "VALUES (?, ?, ?, ?)",
                ("first", "sent_pdf", "sent_estimate_pdf_miner", "shared-key"),
            )
            conn.commit()
            # Sanity: the v1 (evidence_type, source_record_id) index would
            # block the partner today.
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO evidence_ledger "
                    "(evidence_id, evidence_type, source_system, source_record_id) "
                    "VALUES (?, ?, ?, ?)",
                    ("would-collide", "sent_pdf", "ae_dashboard", "shared-key"),
                )
                conn.commit()

        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            # Same insert NOW succeeds — different source_system, same
            # (evidence_type, source_record_id), allowed under v2.
            conn.execute(
                "INSERT INTO evidence_ledger "
                "(evidence_id, evidence_type, source_system, source_record_id) "
                "VALUES (?, ?, ?, ?)",
                ("partner", "sent_pdf", "ae_dashboard", "shared-key"),
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger "
                "WHERE source_record_id = 'shared-key'"
            ).fetchone()[0]
        self.assertEqual(count, 2,
                         "v2 must allow per-source-system namespacing of the "
                         "same source_record_id under the same evidence_type")

    def test_evidence_ledger_preserves_existing_records(self) -> None:
        """Applying ledger upgrade to a DB with seeded rows must not lose
        any of the pre-existing memory or connection data."""
        _create_legacy_schema(self.db)
        seeded_ids = _seed_legacy_records(self.db, count=15)

        with sqlite3.connect(str(self.db)) as conn:
            before_mem = conn.execute(
                "SELECT id, label, content, salience FROM memories ORDER BY id"
            ).fetchall()
            before_conn = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections ORDER BY id"
            ).fetchall()

        SchemaUpgrade(str(self.db)).upgrade()

        with sqlite3.connect(str(self.db)) as conn:
            after_mem = conn.execute(
                "SELECT id, label, content, salience FROM memories ORDER BY id"
            ).fetchall()
            after_conn = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections ORDER BY id"
            ).fetchall()
            ledger_count = conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger"
            ).fetchone()[0]

        self.assertEqual([r[0] for r in after_mem], seeded_ids)
        self.assertEqual(before_mem, after_mem,
                         "ledger migration changed pre-existing memories")
        self.assertEqual(before_conn, after_conn,
                         "ledger migration changed pre-existing connections")
        self.assertEqual(ledger_count, 0,
                         "freshly-created ledger must be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
