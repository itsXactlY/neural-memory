"""Idempotent additive schema upgrade for Phase 7 unified-graph fields.

Extends the SCHEMA + _migrate_bitemporal pattern from memory_client.py to add
typed node/edge fields, bi-temporal validity, salience instrumentation, A-MEM
metadata, governance fields, and locus overlay support — all as nullable
ALTER TABLE additions, no row rewrites, no destructive changes.

Pattern: PRAGMA table_info + ALTER TABLE ADD COLUMN with skip-if-exists.
Calling upgrade() repeatedly is a no-op after the first successful run.

Usage:
    SchemaUpgrade("/path/to/memory.db").upgrade()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# memories: 16 new columns. None exist on legacy 8-col schema.
_MEMORY_COLUMNS: list[tuple[str, str]] = [
    ("kind",                    "TEXT DEFAULT 'unknown'"),
    ("confidence",              "REAL DEFAULT 1.0"),
    ("valid_from",              "REAL DEFAULT NULL"),
    ("valid_to",                "REAL DEFAULT NULL"),
    ("transaction_time",        "REAL DEFAULT NULL"),
    ("origin_system",           "TEXT DEFAULT 'neural_memory'"),
    ("source",                  "TEXT DEFAULT NULL"),
    ("metadata_json",           "TEXT DEFAULT NULL"),
    ("memory_visibility",       "TEXT DEFAULT 'internal'"),
    ("pin_state",               "TEXT DEFAULT 'normal'"),
    ("decay_rate",              "REAL DEFAULT NULL"),
    ("reuse_count",             "INTEGER DEFAULT 0"),
    ("last_reinforced_at",      "REAL DEFAULT NULL"),
    ("extracted_entities_json", "TEXT DEFAULT NULL"),
    ("locus_id",                "INTEGER DEFAULT NULL"),
    ("procedural_score",        "REAL DEFAULT NULL"),
]

# connections: 10 spec columns. edge_type, valid_from, valid_to are typically
# already present from earlier _migrate_bitemporal/install_database; the
# idempotent guard skips them. transaction_time is added alongside the
# existing ingestion_time column (Phase 7 spec uses transaction_time as the
# canonical name; ingestion_time data is preserved for backward compat).
_CONNECTION_COLUMNS: list[tuple[str, str]] = [
    ("edge_type",            "TEXT DEFAULT 'similar'"),
    ("confidence",           "REAL DEFAULT 1.0"),
    ("valid_from",           "REAL DEFAULT NULL"),
    ("valid_to",             "REAL DEFAULT NULL"),
    ("transaction_time",     "REAL DEFAULT NULL"),
    ("origin_system",        "TEXT DEFAULT 'neural_memory'"),
    ("salience",             "REAL DEFAULT 1.0"),
    ("last_strengthened_at", "REAL DEFAULT NULL"),
    ("evidence_count",       "INTEGER DEFAULT 0"),
    ("metadata_json",        "TEXT DEFAULT NULL"),
]


class SchemaUpgrade:
    """Idempotent additive schema upgrade.

    Safe to call repeatedly. Existing columns are not redefined; existing
    rows are preserved unchanged.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    @staticmethod
    def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    @classmethod
    def _add_columns(
        cls,
        conn: sqlite3.Connection,
        table: str,
        cols: list[tuple[str, str]],
    ) -> int:
        existing = cls._existing_columns(conn, table)
        added = 0
        for name, decl in cols:
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                added += 1
            except sqlite3.OperationalError:
                # Race with concurrent migration or unsupported on this sqlite; skip silently.
                pass
        return added

    @staticmethod
    def _ensure_fts5(conn: sqlite3.Connection) -> int:
        """Phase 7 Commit 5: create memories_fts virtual table if missing,
        backfill missing user memories, and DEFENSIVELY remove entity rows
        that may have snuck in via stale-code paths (e.g., a long-running
        process holding pre-fix memory_client.py in memory).

        Returns number of rows backfilled (positive) minus entity rows
        cleaned (negative contribution). Idempotent.

        Internal-content mode: FTS index stores content. Trade-off: 2x
        storage for content text, in exchange for trivial sync. Negligible
        at AE scale.

        Silent no-op if SQLite was compiled without FTS5 support.
        """
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content)"
            )
        except sqlite3.OperationalError:
            return 0

        # Defensive cleanup: remove entity rows from FTS index. Triggers
        # whenever a long-running process with stale code has inserted entity
        # content (caught 2026-05-01 — bcd72db was a forward-guard only;
        # running hermes kept adding entity rows until it reloaded the module).
        try:
            conn.execute(
                """DELETE FROM memories_fts
                   WHERE rowid IN (SELECT id FROM memories WHERE kind = 'entity')"""
            )
        except sqlite3.OperationalError:
            pass

        # Backfill: only memories whose rowid is missing from the FTS index.
        # Skip kind='entity' rows (entities don't belong in sparse search).
        cur = conn.execute(
            """INSERT INTO memories_fts(rowid, content)
               SELECT m.id, m.content FROM memories m
               WHERE NOT EXISTS (SELECT 1 FROM memories_fts f WHERE f.rowid = m.id)
               AND m.content IS NOT NULL
               AND (m.kind IS NULL OR m.kind != 'entity')"""
        )
        return cur.rowcount or 0

    def upgrade(self) -> dict[str, int]:
        """Apply all pending additive migrations. Returns stats dict."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            mem_added = self._add_columns(conn, "memories", _MEMORY_COLUMNS)
            conn_added = self._add_columns(conn, "connections", _CONNECTION_COLUMNS)
            fts_backfilled = self._ensure_fts5(conn)
            conn.commit()
        return {
            "memories_columns_added": mem_added,
            "connections_columns_added": conn_added,
            "fts_rows_backfilled": fts_backfilled,
        }


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".neural_memory" / "memory.db")
    result = SchemaUpgrade(db).upgrade()
    print(f"Schema upgrade complete on {db}: {result}")
