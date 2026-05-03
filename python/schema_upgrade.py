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


# -----------------------------------------------------------------------------
# Evidence ledger (S2 packet 2026-05-03)
# -----------------------------------------------------------------------------
# Additive SQLite-backed identity authority for AEEvidenceIngest. Replaces the
# JSON-scan dedup in record_evidence_artifact with a real UNIQUE-constrained
# ledger row. Keyed by deterministic evidence_id (sha256 of
# (evidence_type, source_system, source_record_id), 16 hex chars). Two unique
# indexes guard against the same source row being inserted under conflicting
# evidence_id derivations.
#
# user_version protocol: legacy DBs ship at 0; this migration bumps to 1.
# Future schema migrations bump by 1 each — read current value before bumping.
# Idempotent: applying an already-current DB is a no-op.
_EVIDENCE_LEDGER_TARGET_USER_VERSION = 1

_EVIDENCE_LEDGER_DDL = """
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
)
"""

_EVIDENCE_LEDGER_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS "
    "idx_evidence_ledger_source ON evidence_ledger (source_system, source_record_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS "
    "idx_evidence_ledger_type_record ON evidence_ledger (evidence_type, source_record_id)",
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

    @staticmethod
    def _ensure_evidence_ledger(conn: sqlite3.Connection) -> dict[str, int]:
        """S2 packet 2026-05-03: idempotent additive evidence ledger.

        Reads PRAGMA user_version. If already at or past target, no-op
        (returns zero counters). Otherwise creates evidence_ledger table +
        UNIQUE indexes and bumps user_version to target.

        Additive only — no DROP/ALTER on existing tables. Safe to apply
        repeatedly; safe to apply on a DB that already has the ledger
        (CREATE IF NOT EXISTS handles partial-prior-runs cleanly).
        """
        current_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version >= _EVIDENCE_LEDGER_TARGET_USER_VERSION:
            return {
                "ledger_created": 0,
                "ledger_indexes_created": 0,
                "user_version_before": current_version,
                "user_version_after": current_version,
            }

        # Table existence pre-check so we can report whether THIS run created
        # it (vs a partial prior run). CREATE IF NOT EXISTS still runs either
        # way — this is informational only.
        existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_ledger'"
        ).fetchone() is not None

        conn.execute(_EVIDENCE_LEDGER_DDL)
        for ddl in _EVIDENCE_LEDGER_INDEXES:
            conn.execute(ddl)

        # Bump user_version. PRAGMA user_version doesn't accept '?' binding;
        # the value is a class-level constant so f-string is safe.
        conn.execute(
            f"PRAGMA user_version = {_EVIDENCE_LEDGER_TARGET_USER_VERSION}"
        )

        return {
            "ledger_created": 0 if existed else 1,
            "ledger_indexes_created": len(_EVIDENCE_LEDGER_INDEXES),
            "user_version_before": current_version,
            "user_version_after": _EVIDENCE_LEDGER_TARGET_USER_VERSION,
        }

    def upgrade(self) -> dict[str, int]:
        """Apply all pending additive migrations. Returns stats dict."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            mem_added = self._add_columns(conn, "memories", _MEMORY_COLUMNS)
            conn_added = self._add_columns(conn, "connections", _CONNECTION_COLUMNS)
            fts_backfilled = self._ensure_fts5(conn)
            ledger_stats = self._ensure_evidence_ledger(conn)
            conn.commit()
        return {
            "memories_columns_added": mem_added,
            "connections_columns_added": conn_added,
            "fts_rows_backfilled": fts_backfilled,
            **ledger_stats,
        }


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".neural_memory" / "memory.db")
    result = SchemaUpgrade(db).upgrade()
    print(f"Schema upgrade complete on {db}: {result}")
