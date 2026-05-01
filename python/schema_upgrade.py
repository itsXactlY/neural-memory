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

    def upgrade(self) -> dict[str, int]:
        """Apply all pending additive migrations. Returns counts of cols added per table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            mem_added = self._add_columns(conn, "memories", _MEMORY_COLUMNS)
            conn_added = self._add_columns(conn, "connections", _CONNECTION_COLUMNS)
            conn.commit()
        return {
            "memories_columns_added": mem_added,
            "connections_columns_added": conn_added,
        }


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".neural_memory" / "memory.db")
    result = SchemaUpgrade(db).upgrade()
    print(f"Schema upgrade complete on {db}: {result}")
