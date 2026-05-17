#!/usr/bin/env python3
"""memory_client.py - Python client for Mazemaker Adapter.

This is the hot path: SQLite persistence, optional C++/GPU/HNSW indexes,
hybrid retrieval, typed temporal graph edges, and PPR thinking.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

from license import has_feature  # Pro feature gate; community → False

logger = logging.getLogger(__name__)


# ============================================================================
# Find the shared library
# ============================================================================

from _lib_finder import find_lib as _find_lib  # canonical resolver


# ============================================================================
# C API struct definitions
# ============================================================================

class CSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 4096),
    ]


class CStats(ctypes.Structure):
    _fields_ = [
        ("total_stores", ctypes.c_uint64),
        ("total_retrieves", ctypes.c_uint64),
        ("total_searches", ctypes.c_uint64),
        ("total_consolidations", ctypes.c_uint64),
        ("avg_store_us", ctypes.c_uint64),
        ("avg_retrieve_us", ctypes.c_uint64),
        ("graph_nodes", ctypes.c_size_t),
        ("graph_edges", ctypes.c_size_t),
        ("hopfield_patterns", ctypes.c_size_t),
        ("hopfield_occupancy", ctypes.c_float),
    ]


# ============================================================================
# SQLite persistence layer
# ============================================================================

DB_PATH = Path.home() / ".mazemaker" / "engine" / "memory.db"


def _resolve_hf_snapshot(model_name: str) -> Optional[str]:
    """Return a local snapshot directory for model_name, or None if not cached.

    Checks ~/.mazemaker/engine/models first, then the default HF hub cache.
    Passing a snapshot path directly to SentenceTransformer / CrossEncoder
    avoids any network contact with the HF Hub.
    """
    safe_name = model_name.replace("/", "--")
    search_dirs = [
        Path.home() / ".mazemaker" / "engine" / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for base in search_dirs:
        cache_base = base / f"models--{safe_name}"
        refs_main = cache_base / "refs" / "main"
        if refs_main.exists():
            snap_hash = refs_main.read_text().strip()
            snap = cache_base / "snapshots" / snap_hash
            if snap.exists() and (snap / "config.json").exists():
                return str(snap)
        snapshots_dir = cache_base / "snapshots"
        if snapshots_dir.exists():
            for snap in sorted(snapshots_dir.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    return str(snap)
    return None

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT (unixepoch()),
    last_accessed REAL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    edge_type TEXT DEFAULT 'similar',
    created_at REAL DEFAULT (unixepoch()),
    event_time REAL,
    ingestion_time REAL DEFAULT (unixepoch()),
    valid_from REAL,
    valid_to REAL,
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS memory_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason TEXT DEFAULT 'conflict_fusion',
    created_at REAL DEFAULT (unixepoch()),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_label ON memories(label);
CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_id);
CREATE INDEX IF NOT EXISTS idx_connections_pair ON connections(source_id, target_id);
CREATE INDEX IF NOT EXISTS idx_connections_edge_type_weight ON connections(edge_type, weight);
CREATE INDEX IF NOT EXISTS idx_memory_revisions_memory ON memory_revisions(memory_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    label,
    content,
    content='memories',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, label, content) VALUES (new.id, new.label, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, label, content)
    VALUES('delete', old.id, old.label, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF label, content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, label, content)
    VALUES('delete', old.id, old.label, old.content);
    INSERT INTO memories_fts(rowid, label, content) VALUES (new.id, new.label, new.content);
END;
"""


class SQLiteStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA wal_autocheckpoint=100000")
        # Wait up to 30s on a busy DB rather than failing fast.  When an
        # external dream_worker is running concurrent NREM/REM batch
        # transactions, every Mazemaker.__init__ from the mcp container
        # hits _ensure_schema_extensions which writes — without this
        # busy_timeout, a single overlapping write window throws
        # "database is locked", which the architect dashboard renders
        # as a generic Internal-error tile.
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(SCHEMA)
        self._ensure_schema_extensions()
        self._fts_available = self._ensure_fts()
        self.conn.commit()
        self._lock = threading.Lock()
        # Stop event for bg_checkpoint — without it, close() would leave the
        # thread running forever (waking every 60s to call .execute on a
        # closed connection, eat the exception, sleep again). With many
        # short-lived SQLiteStore instances (test runs, tools), that's a
        # thread-leak per close. Daemon=True only saves us at process exit.
        self._stop_checkpoint = threading.Event()
        self._checkpoint_thread = threading.Thread(
            target=self._bg_checkpoint, daemon=True, name="sqlite-wal-checkpoint",
        )
        self._checkpoint_thread.start()

    def get_meta(self, key: str) -> Optional[str]:
        # set_meta holds self._lock; get_meta must too — without it, a
        # reader can observe a half-applied write on the same connection
        # (sqlite3.Connection is *not* thread-safe at this level — the
        # check_same_thread=False option only suppresses the assertion).
        with self._lock:
            row = self.conn.execute("SELECT value FROM db_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO db_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def _ensure_schema_extensions(self) -> None:
        """Idempotent migrations for existing DBs."""
        # Tiny key/value table used for embedding fingerprint and other
        # invariants the codebase needs to enforce one-model-per-DB.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS db_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        # ColBERT-style late-interaction token cache. Optional/nullable —
        # existing memories left as NULL; recall gracefully skips this
        # channel for any memory missing the blob. See colbert_helper.py
        # for the design and byte layout. ~64 KB per populated row at 32
        # tokens × 1024 dims fp16; opt-in only via MM_COLBERT_ENABLED=1
        # because 230k memories = 14.7 GB extra disk.
        mem_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "colbert_tokens" not in mem_cols:
            try:
                self.conn.execute("ALTER TABLE memories ADD COLUMN colbert_tokens BLOB")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(connections)").fetchall()}
        migrations = {
            "event_time": "ALTER TABLE connections ADD COLUMN event_time REAL",
            # SQLite cannot ADD COLUMN with DEFAULT (unixepoch()) on an existing
            # table; add plain REAL then backfill below.
            "ingestion_time": "ALTER TABLE connections ADD COLUMN ingestion_time REAL",
            "valid_from": "ALTER TABLE connections ADD COLUMN valid_from REAL",
            "valid_to": "ALTER TABLE connections ADD COLUMN valid_to REAL",
        }
        for col, sql in migrations.items():
            if col not in cols:
                try:
                    self.conn.execute(sql)
                except sqlite3.OperationalError as exc:
                    # Duplicate-column races are harmless; every other schema
                    # error must surface instead of leaving a half-migrated DB.
                    if "duplicate column" not in str(exc).lower():
                        raise
        self.conn.execute("UPDATE connections SET ingestion_time = COALESCE(ingestion_time, created_at, unixepoch())")
        self.conn.execute("UPDATE connections SET valid_from = COALESCE(valid_from, event_time, created_at)")
        self.conn.execute("UPDATE connections SET edge_type = 'similar' WHERE edge_type IS NULL OR edge_type = ''")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_connections_valid_time ON connections(valid_from, valid_to)")

        # Dream-engine schema was historically created lazily by the
        # SQLiteDreamBackend the first time `mazemaker_dream` ran. On a
        # fresh customer pod that's never run a dream cycle, the
        # dream_sessions / connection_history / dream_insights tables
        # don't exist — and the wonderland `/architect/manifest` +
        # `/dream/*` endpoints 500'd as soon as the dashboard loaded.
        # Run the dream schema as part of store init so it's there from
        # minute 1, before any dashboard call.
        try:
            from dream_engine import _DREAM_SCHEMA  # type: ignore[import]
            self.conn.executescript(_DREAM_SCHEMA)
        except Exception:
            # If dream_engine isn't importable for any reason, fall
            # through silently — wonderland endpoints have a defensive
            # `try ... OperationalError → []` fallback for this case.
            pass

        # connection_history.dream_session_id — added 2026-05-06 so the
        # /dream/cycles/{id}/diff endpoint can resolve via FK instead of a
        # timestamp-window scan. Legacy rows stay NULL; new dream-engine
        # writes populate it. Index for the diff lookup.
        try:
            ch_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(connection_history)").fetchall()}
            if ch_cols and "dream_session_id" not in ch_cols:
                self.conn.execute("ALTER TABLE connection_history ADD COLUMN dream_session_id INTEGER")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conn_history_session "
                "ON connection_history(dream_session_id)"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

        # Canonicalise legacy connection rows. Iter 23 made every new write
        # satisfy source<target, but rows that pre-date that fix (or that
        # came in via paths that bypassed add_connection) can still carry
        # arbitrary orientation. Without this sweep, `WHERE source_id=? AND
        # target_id=?` lookups miss the legacy rows half the time and
        # downstream code (recall connection rendering, in-memory graph
        # hydration) sees inconsistent state.
        try:
            # Step 1: when a non-canonical row has a canonical twin, lift the
            # canonical row's weight to MAX(both) so we don't lose the
            # potentially-higher weight from the row we're about to drop.
            # add_connection's MERGE policy is \"keep the larger weight\", so
            # the migration must preserve that semantic.
            self.conn.execute("""
                UPDATE connections
                   SET weight = (
                       SELECT MAX(c1.weight, c2.weight)
                         FROM connections c1, connections c2
                        WHERE c1.id = connections.id
                          AND c2.source_id = connections.target_id
                          AND c2.target_id = connections.source_id
                   )
                 WHERE source_id < target_id
                   AND EXISTS (
                       SELECT 1 FROM connections c2
                        WHERE c2.source_id = connections.target_id
                          AND c2.target_id = connections.source_id
                   )
            """)
            # Step 2: drop the non-canonical duplicate.
            #
            # We capture rowcount + emit an INFO log on non-zero so the
            # mass-deletes from this sweep stop being silent. Previously a
            # restart could delete 1M+ legacy duplicates and the only signal
            # was a sudden drop in `connections` count between two reads —
            # nothing in dream_sessions or any log line accounted for it.
            cur = self.conn.execute("""
                DELETE FROM connections
                 WHERE source_id > target_id
                   AND EXISTS (
                       SELECT 1 FROM connections c2
                        WHERE c2.source_id = connections.target_id
                          AND c2.target_id = connections.source_id
                   )
            """)
            canon_dups = int(cur.rowcount or 0)
            # Step 3: swap survivors in place so they're canonical going forward.
            cur = self.conn.execute("""
                UPDATE connections
                   SET source_id = target_id, target_id = source_id
                 WHERE source_id > target_id
            """)
            canon_swaps = int(cur.rowcount or 0)
            if canon_dups or canon_swaps:
                logger.info(
                    "canonicalisation sweep: deleted %d non-canonical duplicates, "
                    "swapped %d non-canonical singletons (one-shot legacy migration; "
                    "no-op once the table is fully canonical)",
                    canon_dups, canon_swaps,
                )
        except sqlite3.OperationalError:
            # Schema oddity (very old DB without these columns) — leave as-is;
            # subsequent writes are canonical regardless.
            pass

    def _ensure_fts(self) -> bool:
        try:
            self.conn.executescript(FTS_SCHEMA)
            mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            fts_count = self.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            if mem_count and fts_count == 0:
                self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            return True
        except sqlite3.OperationalError as exc:
            # FTS5 disabled in this SQLite build — search will use the
            # capped lexical-overlap fallback. Log once so operators don't
            # mistake the fallback's lower recall for a regression.
            logger.warning("FTS5 unavailable (%s) — search_bm25 will use lexical fallback", exc)
            return False

    def _connect_reader(self) -> sqlite3.Connection:
        # Reader connections must agree with the writer on journal mode and
        # busy_timeout. A bare sqlite3.connect() uses defaults (no WAL,
        # 5 s busy_timeout) and routinely hits "database is locked" on a
        # WAL-mode DB under concurrent writes.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _bg_checkpoint(self):
        # Wait on the stop event instead of plain time.sleep so close() can
        # signal an immediate exit. wait(60) returns True if the event was
        # set during the wait — that's our shutdown signal.
        while not self._stop_checkpoint.wait(60):
            try:
                with self._lock:
                    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            except sqlite3.Error as exc:
                # On a closed connection (post-close race) this raises;
                # the stop event will fire next iteration and break us out.
                # Narrow from bare-except so we still get a stack on
                # genuinely-unexpected errors (e.g. disk full → WAL
                # would otherwise grow without bound).
                if not self._stop_checkpoint.is_set():
                    logger.warning("wal_checkpoint failed: %s", exc)

    @staticmethod
    def _unpack_embedding(blob: bytes | None) -> list[float]:
        if not blob:
            return []
        dim = len(blob) // 4
        return list(struct.unpack(f"{dim}f", blob))

    # FTS5 stopwords — natural-language scaffolding tokens that bloat the
    # AND-form of multi-word queries until BM25 returns 0 hits despite
    # the corpus containing the answer. Cherry-picked from upstream PR #5
    # (commit 2d9b5b9): the AE-domain bench saw 0/240 sparse hits before
    # this filter, 240/240 after.
    _FTS_STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "of", "in", "on", "at", "to", "for", "by",
        "with", "from", "as", "this", "that", "these", "those", "it",
        "its", "what", "when", "where", "why", "how", "who", "which",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "should", "could", "can", "may", "might", "i", "we", "you",
        "they", "he", "she", "find", "show", "list", "give", "tell",
        "memories", "mention", "memory", "notes", "note", "about",
    })

    @staticmethod
    def _sanitize_fts_query(query: str, mode: str = "and") -> str:
        # Preserve slash + hyphen inside tokens ("12/2 romex", "lot-27").
        # Lowercase for stopword matching; FTS5 is case-insensitive anyway.
        raw = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_/\-]{1,}", (query or "").lower())
        tokens = []
        for t in raw[:24]:
            t = t.replace('"', '')
            # Need at least 2 alphanumerics so "1-2" / "a-" don't slip through.
            if len(re.sub(r"[^A-Za-z0-9]", "", t)) < 2:
                continue
            if t in SQLiteStore._FTS_STOPWORDS:
                continue
            tokens.append(t)
            if len(tokens) >= 12:
                break
        if not tokens:
            return ""
        op = " OR " if mode == "or" else " "
        return op.join(f'"{t}"' for t in tokens)

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        text = text or ""
        entities: list[str] = []
        # CamelCase / ProperCase / ALLCAPS / token-with-digits are useful for project names.
        for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", text):
            if tok[0].isupper() or any(ch.isdigit() for ch in tok) or re.search(r"[a-z][A-Z]", tok):
                entities.append(tok)
        # Quoted phrases are entities too.
        for phrase in re.findall(r"['\"]([^'\"]{3,80})['\"]", text):
            entities.append(phrase.strip())
        # Stable de-dupe.
        seen = set()
        out = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories (label, content, embedding, created_at, last_accessed) VALUES (?, ?, ?, ?, ?)",
                (label, content, blob, now, now),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def remember_batch(self, rows: list[dict]) -> list[int]:
        if not rows:
            return []
        now = time.time()
        params: list[tuple] = []
        for r in rows:
            emb = r.get("embedding") or []
            blob = struct.pack(f"{len(emb)}f", *emb)
            salience = r.get("salience")
            params.append((
                r.get("label"),
                r.get("content", ""),
                blob,
                float(salience) if salience is not None else 1.0,
                now,
                now,
            ))
        # RETURNING preserves input order on SQLite >= 3.35 and avoids the
        # lastrowid+len trick which is brittle under concurrent writers.
        sql = (
            "INSERT INTO memories "
            "(label, content, embedding, salience, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING id"
        )
        ids: list[int] = []
        with self._lock:
            try:
                for p in params:
                    cur = self.conn.execute(sql, p)
                    row = cur.fetchone()
                    ids.append(int(row[0] if not isinstance(row, sqlite3.Row) else row["id"]))
                self.conn.commit()
            except sqlite3.OperationalError:
                # RETURNING unavailable on ancient SQLite — fall back to
                # executemany + lastrowid arithmetic. The FTS AFTER INSERT
                # trigger keeps memories_fts in sync either way.
                self.conn.rollback()
                cur = self.conn.executemany(
                    "INSERT INTO memories "
                    "(label, content, embedding, salience, created_at, last_accessed) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    params,
                )
                last = int(cur.lastrowid or 0)
                n = len(params)
                ids = list(range(last - n + 1, last + 1))
                self.conn.commit()
        return ids

    def get_all(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count FROM memories ORDER BY id"
            ).fetchall()
        results = []
        for r in rows:
            emb = self._unpack_embedding(r["embedding"])
            if not emb:
                continue
            results.append({
                "id": r["id"],
                "label": r["label"],
                "content": r["content"],
                "embedding": emb,
                "salience": r["salience"],
                "created_at": r["created_at"],
                "last_accessed": r["last_accessed"],
                "access_count": r["access_count"],
            })
        return results

    def get(self, id_: int, include_embedding: bool = True) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count FROM memories WHERE id = ?",
            (id_,),
        ).fetchone()
        if not row:
            return None
        out = {
            "id": row["id"],
            "label": row["label"],
            "content": row["content"],
            "salience": row["salience"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
        }
        if include_embedding:
            out["embedding"] = self._unpack_embedding(row["embedding"])
        return out

    def get_many(self, ids: list[int], include_embedding: bool = False) -> dict[int, dict]:
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return {}
        cols = "id, label, content, salience, created_at, last_accessed, access_count"
        if include_embedding:
            cols += ", embedding"
        out: dict[int, dict] = {}
        # Chunk to stay under SQLite's 999-parameter cap. The previous
        # f-string-with-N-placeholders form raised OperationalError
        # whenever a caller passed >= 1000 ids (which the dream engine
        # batch paths and the dashboard can both legitimately do).
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT {cols} FROM memories WHERE id IN ({placeholders})", tuple(chunk),
            ).fetchall()
            for r in rows:
                item = {
                    "id": r["id"],
                    "label": r["label"],
                    "content": r["content"],
                    "salience": r["salience"],
                    "created_at": r["created_at"],
                    "last_accessed": r["last_accessed"],
                    "access_count": r["access_count"],
                }
                if include_embedding:
                    item["embedding"] = self._unpack_embedding(r["embedding"])
                out[item["id"]] = item
        return out

    def find_by_label(self, label: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count FROM memories WHERE label = ? ORDER BY id",
            (label,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"], "label": r["label"], "content": r["content"],
                "embedding": self._unpack_embedding(r["embedding"]),
                "salience": r["salience"], "created_at": r["created_at"],
                "last_accessed": r["last_accessed"], "access_count": r["access_count"],
            })
        return out

    def add_revision(self, memory_id: int, old_content: str, new_content: str, reason: str = "conflict_fusion") -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory_revisions (memory_id, old_content, new_content, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, old_content, new_content, reason, time.time()),
            )
            self.conn.commit()

    def update_memory(self, memory_id: int, content: str, embedding: list[float], label: Optional[str] = None) -> None:
        """Rewrite a memory's content/embedding (e.g. conflict fusion).

        Updates DO NOT bump access_count — that counter tracks how often a
        memory is RECALLED, and is what feeds the salience boost in touch().
        Counting writes there meant every conflict-fused memory immediately
        looked "frequently accessed" and floated to the top of subsequent
        recalls regardless of whether anyone had actually queried it.
        last_accessed is also kept stable for the same reason; touch() owns
        the access timeline.
        """
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        with self._lock:
            if label is None:
                self.conn.execute(
                    "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                    (content, blob, memory_id),
                )
            else:
                self.conn.execute(
                    "UPDATE memories SET label = ?, content = ?, embedding = ? WHERE id = ?",
                    (label, content, blob, memory_id),
                )
            self.conn.commit()

    # ---- ColBERT token cache ------------------------------------------------
    def set_colbert_tokens(self, memory_id: int, blob: Optional[bytes]) -> None:
        """Set or clear the colbert token blob for a memory id."""
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET colbert_tokens = ? WHERE id = ?",
                (blob, int(memory_id)),
            )
            self.conn.commit()

    def get_colbert_tokens_many(self, ids: "list[int]") -> "dict[int, bytes]":
        """Bulk-fetch colbert blobs for the given ids. Missing/null entries
        are simply absent from the returned dict (caller must handle)."""
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return {}
        out: dict[int, bytes] = {}
        # SQLite's 999-parameter cap — chunk for safety on large rerank windows.
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT id, colbert_tokens FROM memories "
                f"WHERE id IN ({placeholders}) AND colbert_tokens IS NOT NULL",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                blob = r["colbert_tokens"]
                if blob:
                    out[int(r["id"])] = bytes(blob)
        return out

    def ensure_dae_schema(self) -> bool:
        """Create the memory_dae_embeddings table if missing. Idempotent.

        Returns True when the table now exists (or already did), False
        when the license gate refuses DAE. Mirrors the engine-side
        contract previously implemented in dae.ensure_schema(conn).
        """
        try:
            from dae import ensure_schema as _ensure
        except Exception:
            return False
        with self._lock:
            return _ensure(self.conn)

    def upsert_dae_vectors(
        self,
        rows: "list[tuple[int, bytes, float, int, int, float]]",
    ) -> int:
        """Bulk INSERT-OR-REPLACE for DAE rows.

        Each row: (memory_id, vector_blob, self_weight, neighbour_k,
        schema_version, computed_at). The previous dae_bulk_compute
        ran this SQL inline against self.store.conn — SQLite-only
        (`INSERT OR REPLACE`, `?` placeholders). Moving it to the
        store keeps the compute path backend-agnostic.
        """
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO memory_dae_embeddings "
                "(memory_id, vector, self_weight, neighbour_k, "
                " schema_version, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def prune_memories_by_label_prefix(self, prefix: str, older_than_ts: float) -> int:
        """Delete memories whose label starts with `prefix` and whose
        created_at is older than `older_than_ts`. Returns the row count.

        Lives on the store rather than as raw SQL in dream_engine so
        the same call works on PostgresStore — the previous form
        (`store.conn.execute("... LIKE ?...")`) was SQLite-only and
        silently failed when dream cycles ran against the PG backend
        (no derived-cluster TTL pruning on PG).
        """
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM memories WHERE label LIKE ? AND created_at < ?",
                (prefix + "%", float(older_than_ts)),
            )
            n = int(cur.rowcount or 0)
            self.conn.commit()
        return n

    def fetch_dae_vectors(self, ids: "list[int]") -> "dict[int, list[float]]":
        """Backend-agnostic shim used by _dae_score_candidates.

        Delegates to dae.fetch_dae_vectors with the SQLite connection
        (the historical implementation). PostgresStore has its own
        method with PG-flavoured SQL; the dispatch happens in
        Mazemaker._dae_score_candidates via getattr on the store.
        """
        try:
            from dae import fetch_dae_vectors as _fetch
        except Exception:
            return {}
        return _fetch(self.conn, list(ids))

    def stream_missing_colbert(self, batch_size: int = 1000, start_after_id: int = 0):
        """Yield (id, content) pairs for memories that don't yet have a
        colbert blob. Used by the migration script. Streams in id order
        so the caller can persist a checkpoint after each batch."""
        last_id = int(start_after_id or 0)
        while True:
            rows = self.conn.execute(
                "SELECT id, content FROM memories "
                "WHERE id > ? AND colbert_tokens IS NULL "
                "ORDER BY id ASC LIMIT ?",
                (last_id, int(batch_size)),
            ).fetchall()
            if not rows:
                return
            for r in rows:
                yield int(r["id"]), (r["content"] or "")
            last_id = int(rows[-1]["id"])

    def stream_long_memories_for_afe(self, *, min_len: int = 500,
                                     limit: int = 1000,
                                     exclude_label_pattern: str = "%::afe::%",
                                     exclude_ids: "set[int] | None" = None):
        """Mirror of PostgresStore.stream_long_memories_for_afe for the
        SQLite backend. Same yield contract: (id, label, content) ordered
        by content length DESC, with already-AFE'd rows filtered out and
        an explicit exclude_ids set honoured in Python."""
        exclude_ids = exclude_ids or set()
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, label, content FROM memories "
                "WHERE length(content) >= ? "
                "  AND (label IS NULL OR label NOT LIKE ?) "
                "ORDER BY length(content) DESC LIMIT ?",
                (int(min_len), exclude_label_pattern, int(limit) * 2),
            ).fetchall()
        kept = 0
        for r in rows:
            mid = int(r["id"])
            if mid in exclude_ids:
                continue
            yield mid, (r["label"] or ""), (r["content"] or "")
            kept += 1
            if kept >= limit:
                return

    def touch(self, id_: int):
        with self._lock:
            row = self.conn.execute(
                "SELECT salience, last_accessed, access_count, (SELECT COUNT(*) FROM connections WHERE source_id = ? OR target_id = ?) AS conn_count FROM memories WHERE id = ?",
                (id_, id_, id_),
            ).fetchone()
            if not row:
                return
            now = time.time()
            old_salience = float(row["salience"] or 1.0)
            last = float(row["last_accessed"] or now)
            idle_days = max(0.0, (now - last) / 86400.0)
            # Was hardcoded -0.03; the configured self._salience_decay_k
            # (used elsewhere by _recompute_salience) was silently ignored
            # on the touch() path, so the knob was inconsistent.
            decayed = old_salience * math.exp(-self._salience_decay_k * idle_days)
            access_count = int(row["access_count"] or 0) + 1
            conn_count = int(row["conn_count"] or 0)
            salience = decayed + 0.04 * math.log1p(access_count) + 0.02 * math.log1p(conn_count)
            salience = max(0.1, min(3.0, salience))
            self.conn.execute(
                "UPDATE memories SET last_accessed = ?, access_count = ?, salience = ? WHERE id = ?",
                (now, access_count, salience, id_),
            )
            self.conn.commit()

    def add_connection(
        self,
        source: int,
        target: int,
        weight: float,
        edge_type: str = "similar",
        event_time: Optional[float] = None,
        ingestion_time: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
    ):
        if source == target:
            return
        source, target = (int(source), int(target))
        if source > target:
            source, target = target, source
        # NaN/Inf must never reach the table — Postgres mirror sorts
        # NaN as greater than every real number under ORDER BY DESC,
        # which poisons mazemaker_graph top_edges. Keep the SQLite
        # writer in lockstep so both backends drop the same rows.
        try:
            wf = float(weight)
        except (TypeError, ValueError):
            return
        if not math.isfinite(wf):
            return
        weight = max(0.0, min(1.0, wf))
        now = time.time()
        event_time = event_time if event_time is not None else now
        ingestion_time = ingestion_time if ingestion_time is not None else now
        valid_from = valid_from if valid_from is not None else event_time
        with self._lock:
            row = self.conn.execute(
                "SELECT id, weight FROM connections WHERE source_id = ? AND target_id = ? AND COALESCE(edge_type, 'similar') = ?",
                (source, target, edge_type),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE connections SET weight = ?, event_time = COALESCE(?, event_time), ingestion_time = COALESCE(?, ingestion_time), valid_from = COALESCE(?, valid_from), valid_to = ? WHERE id = ?",
                    (max(weight, float(row["weight"] or 0.0)), event_time, ingestion_time, valid_from, valid_to, row["id"]),
                )
            else:
                self.conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at, event_time, ingestion_time, valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (source, target, weight, edge_type, now, event_time, ingestion_time, valid_from, valid_to),
                )
            self.conn.commit()

    def add_connections_batch(
        self,
        pairs,
        edge_type: str = "similar",
    ) -> int:
        """Bulk-insert undirected weighted edges in a single transaction.

        Replaces N sequential ``add_connection`` calls (each its own commit)
        for hot paths like dream-cycle Insight that emit thousands of
        derived_from edges per cluster.

        ``pairs`` is an iterable of ``(source, target, weight)`` tuples.
        Self-loops are dropped, endpoints are normalised to (low, high)
        order so undirected edges canonicalise, weights are clamped to
        [0, 1]. Existing edges with the same (source, target, edge_type)
        are skipped (duplicate suppression by SELECT-then-diff). Returns
        number of NEW edges actually inserted.

        Performance: ~50× faster than a per-edge loop on SQLite WAL,
        because we batch the work into one fsync rather than one per row.
        """
        normalised: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for s, t, w in pairs:
            si, ti = int(s), int(t)
            if si == ti:
                continue
            if si > ti:
                si, ti = ti, si
            if (si, ti) in seen:
                continue
            # Reject NaN/Inf — see add_connection note. ``max(0, min(1,
            # nan))`` would otherwise silently coerce to 1.0.
            try:
                wf = float(w)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(wf):
                continue
            wf = max(0.0, min(1.0, wf))
            seen.add((si, ti))
            normalised.append((si, ti, wf))

        if not normalised:
            return 0

        now = time.time()
        with self._lock:
            # Discover existing (source_id, target_id) pairs for this edge_type
            # in a single scan via a temp staging table — keeps the diff to
            # one query regardless of pair count.
            self.conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _conn_stage("
                "source_id INTEGER, target_id INTEGER, weight REAL)"
            )
            try:
                self.conn.execute("DELETE FROM _conn_stage")
                self.conn.executemany(
                    "INSERT INTO _conn_stage VALUES (?, ?, ?)",
                    normalised,
                )
                rows = self.conn.execute(
                    "SELECT s.source_id, s.target_id "
                    "  FROM _conn_stage s "
                    " WHERE EXISTS ("
                    "       SELECT 1 FROM connections c "
                    "        WHERE c.source_id = s.source_id "
                    "          AND c.target_id = s.target_id "
                    "          AND COALESCE(c.edge_type, 'similar') = ?)",
                    (edge_type,),
                ).fetchall()
                existing = {(r["source_id"], r["target_id"]) for r in rows}

                to_insert = [
                    (s, t, w, edge_type, now, now, now, now, None)
                    for (s, t, w) in normalised
                    if (s, t) not in existing
                ]
                if to_insert:
                    self.conn.executemany(
                        "INSERT INTO connections "
                        "(source_id, target_id, weight, edge_type, created_at, "
                        " event_time, ingestion_time, valid_from, valid_to) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        to_insert,
                    )
                self.conn.commit()
                return len(to_insert)
            finally:
                # Free the staging rows but keep the temp table cached for
                # the next call — a fresh CREATE on every call is cheap but
                # not free, and DELETE-from-temp is rounding error.
                try:
                    self.conn.execute("DELETE FROM _conn_stage")
                except Exception:
                    pass

    def get_all_connections(self) -> list[dict]:
        """Return every edge as a list — used for bulk graph hydration.

        One SELECT across the whole connections table. Caller groups
        the rows by node id; cheaper than calling get_connections once
        per memory on startup.
        """
        rows = self.conn.execute(
            "SELECT source_id, target_id, weight, edge_type FROM connections"
        ).fetchall()
        return [
            {
                "source": r["source_id"],
                "target": r["target_id"],
                "weight": float(r["weight"] or 0.0),
                "type": r["edge_type"] or "similar",
            }
            for r in rows
        ]

    def get_connections(self, node_id: int, at_time: Optional[float] = None) -> list[dict]:
        params: list[Any] = [node_id, node_id]
        time_filter = ""
        if at_time is not None:
            time_filter = """
               AND COALESCE(valid_from, event_time, created_at, 0) <= ?
               AND (valid_to IS NULL OR valid_to >= ?)
            """
            params.extend([at_time, at_time])
        rows = self.conn.execute(
            f"""SELECT source_id, target_id, weight, edge_type, created_at,
                       event_time, ingestion_time, valid_from, valid_to
                  FROM connections
                 WHERE (source_id = ? OR target_id = ?)
                 {time_filter}
                 ORDER BY weight DESC""",
            tuple(params),
        ).fetchall()
        return [
            {
                "source": r["source_id"],
                "target": r["target_id"],
                "weight": r["weight"],
                "type": r["edge_type"] or "similar",
                "edge_type": r["edge_type"] or "similar",
                "created_at": r["created_at"],
                "event_time": r["event_time"],
                "ingestion_time": r["ingestion_time"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
            }
            for r in rows
        ]

    def search_bm25(self, query: str, limit: int = 50) -> list[dict]:
        # Two-pass FTS5 query (cherry-picked behaviour from upstream PR #5,
        # commit 2d9b5b9): AND-form first (precise), fall back to OR-form
        # when AND produces zero hits. NL queries with even one out-of-corpus
        # token would otherwise return empty — the OR fallback lets BM25
        # rank by how many tokens *did* match.
        if self._fts_available:
            for mode in ("and", "or"):
                fts_query = self._sanitize_fts_query(query, mode=mode)
                if not fts_query:
                    continue
                try:
                    conn = self._connect_reader()
                    try:
                        rows = conn.execute(
                            "SELECT rowid AS id, bm25(memories_fts) AS rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                            (fts_query, limit),
                        ).fetchall()
                        if rows:
                            return [{"id": int(r["id"]), "score": 1.0 / (i + 1), "rank": float(r["rank"]), "channel": "bm25"} for i, r in enumerate(rows)]
                    finally:
                        conn.close()
                except sqlite3.OperationalError:
                    continue
        # Fallback lexical overlap.
        q = set(re.findall(r"\w+", (query or "").lower()))
        if not q:
            return []
        # Cap the corpus scan. FTS5 may be unavailable on some SQLite
        # builds, in which case search_bm25 fell back to scanning every
        # row in `memories` (an O(n) Python word-set diff on a 195k+
        # corpus — multiple seconds per call). The cap below makes the
        # fallback bounded; quality degrades to "most-recent N matched",
        # which is the right trade vs. blocking a search for seconds.
        FALLBACK_SCAN_CAP = 10_000
        scored = []
        scanned = 0
        for m in self.get_all():
            scanned += 1
            if scanned > FALLBACK_SCAN_CAP:
                logger.warning(
                    "search_bm25 fallback truncated at %d rows (FTS5 unavailable, "
                    "corpus larger than scan cap)", FALLBACK_SCAN_CAP,
                )
                break
            words = set(re.findall(r"\w+", ((m.get("label") or "") + " " + (m.get("content") or "")).lower()))
            overlap = len(q & words)
            if overlap:
                scored.append({"id": m["id"], "score": overlap / len(q), "channel": "bm25"})
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def search_entity(self, query: str, limit: int = 50) -> list[dict]:
        entities = self.extract_entities(query)
        if not entities:
            # Long rare-ish tokens function as ad-hoc entities.
            entities = [t for t in re.findall(r"\b[A-Za-z0-9_\-]{6,}\b", query or "")[:5]]
        if not entities:
            return []
        fts_query = " OR ".join(f'"{e.replace(chr(34), "")}"' for e in entities[:8])
        if self._fts_available and fts_query:
            try:
                conn = self._connect_reader()
                try:
                    rows = conn.execute(
                        "SELECT rowid AS id FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                    return [{"id": int(r["id"]), "score": 1.0 / (i + 1), "matched_entities": entities, "channel": "entity"} for i, r in enumerate(rows)]
                finally:
                    conn.close()
            except sqlite3.OperationalError:
                pass
        # Cap fallback scan — same reason as search_bm25 (FTS5 missing
        # means we degrade to a Python word-search; on 195k corpora the
        # uncapped form was a multi-second blocker).
        ENTITY_FALLBACK_SCAN_CAP = 10_000
        scored = []
        scanned = 0
        lowered = [e.lower() for e in entities]
        for m in self.get_all():
            scanned += 1
            if scanned > ENTITY_FALLBACK_SCAN_CAP:
                logger.warning(
                    "search_entity fallback truncated at %d rows (FTS5 unavailable)",
                    ENTITY_FALLBACK_SCAN_CAP,
                )
                break
            text = ((m.get("label") or "") + " " + (m.get("content") or "")).lower()
            hits = [e for e in lowered if e in text]
            if hits:
                scored.append({"id": m["id"], "score": len(hits) / len(lowered), "matched_entities": hits, "channel": "entity"})
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    # Temporal cue table — matched as substrings against the lowercased
    # query. Each row is (cue, window_seconds). Wider than the previous
    # 3-cue set ("today/yesterday/last week"), which left the channel
    # returning a generic "most recent N" pool for every non-temporal
    # query — including "what's the current X?" — and dragged
    # update_tracking on the 10M bench down to 0.508. The new
    # "current/latest/now/recently/updated" cues cover the
    # update-tracking phrasing.
    _TEMPORAL_CUES = (
        # (cue, window_seconds_back_from_now)
        ("today", 86400),
        ("heute", 86400),
        ("yesterday", 2 * 86400),
        ("gestern", 2 * 86400),
        ("last week", 7 * 86400),
        ("letzte woche", 7 * 86400),
        ("this week", 7 * 86400),
        ("diese woche", 7 * 86400),
        ("last month", 30 * 86400),
        ("this month", 30 * 86400),
        # Update-tracking phrasing: when the user asks about *current*
        # state of something that may have been updated.
        ("latest", 30 * 86400),
        ("newest", 30 * 86400),
        ("most recent", 30 * 86400),
        ("currently", 30 * 86400),
        ("current value", 30 * 86400),
        ("now what", 30 * 86400),
        ("recently", 14 * 86400),
        ("updated", 14 * 86400),
    )

    def search_temporal(self, query: str, limit: int = 50, now: Optional[float] = None) -> list[dict]:
        now = now or time.time()
        q = (query or "").lower()

        # ISO date anchor: YYYY-MM-DD in the query pins a ±1d window.
        import re as _re
        iso_match = _re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
        window_seconds: Optional[float] = None
        anchor_ts: Optional[float] = None
        if iso_match:
            try:
                import datetime as _dt
                y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
                anchor_ts = _dt.datetime(y, m, d).timestamp()
            except (ValueError, OverflowError):
                anchor_ts = None

        if anchor_ts is None:
            for cue, win in self._TEMPORAL_CUES:
                if cue in q:
                    window_seconds = float(win)
                    # "yesterday" / "gestern" semantics: between -2d and -1d.
                    if cue in ("yesterday", "gestern"):
                        where = "WHERE created_at BETWEEN ? AND ?"
                        params = [now - 2 * 86400, now - 86400]
                        break
                    where = "WHERE created_at >= ?"
                    params = [now - window_seconds]
                    break
            else:
                # No temporal cue → empty channel. Was returning
                # "most recent N" regardless of intent, which polluted
                # the RRF fusion with recency bias on every query.
                return []
        else:
            where = "WHERE created_at BETWEEN ? AND ?"
            params = [anchor_ts - 86400, anchor_ts + 86400]

        rows = self.conn.execute(
            f"SELECT id, created_at, last_accessed FROM memories {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params + [limit]),
        ).fetchall()
        return [{"id": int(r["id"]), "score": 1.0 / (i + 1), "created_at": r["created_at"], "channel": "temporal"} for i, r in enumerate(rows)]

    def get_stats(self) -> dict:
        mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn_count = self.conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        rev_count = self.conn.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0]
        return {"memories": mem_count, "connections": conn_count, "revisions": rev_count, "fts": self._fts_available}

    # -- Backend-abstract helpers used by NeuralMemory hot paths ----------
    # These were originally raw self.store.conn.execute(...) calls inside
    # memory_client.py — refactored so the same call sites work against
    # PostgresStore (Pro/Enterprise) without leaking SQLite specifics.

    def get_max_id(self) -> int:
        row = self.conn.execute("SELECT MAX(id) FROM memories").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def recent_semantic_pool(self, limit: int,
                             exclude_label_prefix: Optional[str] = None) -> list[dict]:
        """Top-N newest memories with embedding present, optionally
        excluding labels with a given prefix.

        Returns dicts with id, content, embedding (list[float]).
        """
        if exclude_label_prefix:
            rows = self.conn.execute(
                "SELECT id, content, embedding FROM memories "
                "WHERE embedding IS NOT NULL "
                "  AND (label IS NULL OR label NOT LIKE ?) "
                "ORDER BY created_at DESC LIMIT ?",
                (exclude_label_prefix + "%", int(limit)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, content, embedding FROM memories "
                "WHERE embedding IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": int(r["id"]),
                "content": r["content"],
                "embedding": self._unpack_embedding(r["embedding"]),
            })
        return out

    def weighted_edges(self) -> list[tuple[int, int, float]]:
        """Every edge with positive weight as (source, target, weight)
        tuples — feeds the GPU PPR adjacency build."""
        rows = self.conn.execute(
            "SELECT source_id, target_id, weight FROM connections WHERE weight > 0"
        ).fetchall()
        return [(int(r["source_id"]), int(r["target_id"]), float(r["weight"] or 0.0))
                for r in rows]

    def top_weighted_edges(self, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT source_id, target_id, weight, edge_type FROM connections "
            "ORDER BY weight DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [
            {
                "source": int(r["source_id"]),
                "target": int(r["target_id"]),
                "weight": float(r["weight"] or 0.0),
                "type": r["edge_type"] or "similar",
                "edge_type": r["edge_type"] or "similar",
            }
            for r in rows
        ]

    def prune_connections_below(self, threshold: float) -> int:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM connections WHERE weight < ?", (float(threshold),)
            )
            self.conn.commit()
            deleted = int(cur.rowcount or 0)
        if deleted:
            logger.info("pruned %d connections below weight=%.4f", deleted, float(threshold))
        return deleted

    def close(self):
        # Signal the bg_checkpoint thread to stop and give it a brief window
        # to exit cleanly before we close the connection out from under it.
        # daemon=True still saves us at process exit, but during in-process
        # store lifecycle (tests, tools) we want a clean teardown.
        self._stop_checkpoint.set()
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join(timeout=2.0)
        self.conn.close()


# ============================================================================
# Mazemaker Client
# ============================================================================

class Mazemaker:
    """Python interface to the Mazemaker system."""

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        embedding_backend: str = "auto",
        use_cpp: bool = True,
        embedder=None,
        retrieval_mode: str = "semantic",
        retrieval_candidates: int = 64,
        use_hnsw: bool | str | None = None,
        lazy_graph: bool = False,
        think_engine: str = "bfs",
        rerank: bool = False,
        channel_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        salience_decay_k: float = 0.03,
        ppr_alpha: float = 0.15,
        ppr_iters: int = 20,
        ppr_hops: int = 2,
        mmr_lambda: float = 0.0,
        recall_score_floor: float = 0.0,
        recall_score_percentile: float = 0.0,
    ):
        if embedder is not None:
            self.embedder = embedder
        else:
            from embed_provider import EmbeddingProvider
            self.embedder = EmbeddingProvider(backend=embedding_backend)

        # Backend dispatch. SQLite is the default; MM_DB_BACKEND=postgres
        # opts in to the Postgres + pgvector store IF the license carries
        # the `postgres` feature (Pro/Enterprise tier with backend=postgres
        # claim).  Community / Lite installs fall back to SQLite with a
        # one-line warning so the operator knows what happened.
        backend_choice = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
        if backend_choice == "postgres" and has_feature("postgres"):
            from postgres_store import PostgresStore
            self.store = PostgresStore()
        else:
            if backend_choice == "postgres":
                logger.warning(
                    "MM_DB_BACKEND=postgres requested but the Postgres "
                    "backend is a Pro feature; falling back to SQLite. "
                    "See https://mazemaker.online/#pricing"
                )
            self.store = SQLiteStore(db_path)

        self.dim = self.embedder.dim
        self._db_path = Path(db_path)

        # ONE MODEL invariant: pin the DB to whichever embedding backend
        # wrote its first memory. A subsequent open with a different model
        # would have produced silent dim-truncation in cosine similarity
        # (see iter 11) — surface it loudly instead.
        self._embed_fingerprint = self._compute_embed_fingerprint()
        self._dim_locked, self._dim_mismatch_reason = self._enforce_dim_lock()
        self._retrieval_mode = (retrieval_mode or "semantic").lower()
        self._retrieval_candidates = max(8, int(retrieval_candidates or 64))
        self._lazy_graph = bool(lazy_graph)
        self._think_engine = (think_engine or "bfs").lower()
        self._rerank = bool(rerank)
        self._reranker = None
        self._reranker_failed = False
        self._rrf_k = int(rrf_k or 60)
        self._salience_decay_k = float(salience_decay_k or 0.03)
        self._ppr_alpha = float(ppr_alpha or 0.15)
        self._ppr_iters = int(ppr_iters or 20)
        self._ppr_hops = int(ppr_hops or 2)
        # Maximal Marginal Relevance: trade pure relevance for diversity in
        # the returned top-k. lambda=1.0 = pure relevance (default off);
        # lambda=0.0 = pure diversity. Typical sweet spot is 0.6-0.8.
        self._mmr_lambda = max(0.0, min(1.0, float(mmr_lambda or 0.0)))
        # Hard noise floor: drop any result whose final relevance lands below
        # this. Off by default; set to e.g. 0.15 to refuse to surface garbage
        # when nothing relevant exists, instead of returning weak top-k.
        # NOTE: relevance is RRF-derived and operates in ~[0, 0.05]; the
        # codex 2026-04-28 benchmark caught that values >= 0.2 nuke
        # everything. Use the new `score_percentile` kwarg on recall() for
        # a properly calibrated [0,1] alternative.
        self._recall_score_floor = max(0.0, float(recall_score_floor or 0.0))
        # Constructor-level default for score_percentile. The recall() kwarg
        # of the same name overrides this per-call. Calibrated [0,1] knob —
        # see comment on score_floor above for why this companion exists.
        self._recall_score_percentile = max(0.0, min(1.0, float(recall_score_percentile or 0.0)))
        # Default channel mix. The 2026-04-28 benchmark's channel_ablation
        # measured per-channel contribution on paraphrase data:
        #   ppr      Δmrr=-0.1216 (most helpful)
        #   semantic Δmrr=-0.0416
        #   entity   Δmrr=-0.0216 / Δrecall=-0.04
        #   bm25     null
        #   temporal null
        #   salience Δmrr=+0.0064 (slightly harmful when ON)
        # Defaults preserved for backwards compatibility — but
        # retrieval_mode="lean" zeroes the dead-weight channels.
        self._channel_weights = {
            "semantic": 1.0,
            "bm25": 0.9,
            "entity": 1.0,
            "temporal": 0.35,
            # `recency`: weight 1.2 — slightly above semantic so freshly-saved
            # status updates outrank dense-keyword old memories on "status of X"
            # / "how is Y going" queries. NEVER zeroed (kept active in lean
            # mode) because that's where the channel earns its weight: lean
            # zeros plain `temporal`, leaving recency-aware retrieval to this
            # channel alone. See _recent_semantic_candidates.
            "recency": 1.2,
            "ppr": 0.55,
            "salience": 0.25,
        }
        # Lean preset: zeroes the channels the 2026-04-28 benchmark proved
        # are dead-weight or actively harmful (bm25, temporal, salience).
        # On synthetic data: 4× p50 speedup at -0.02 recall vs skynet.
        # On real prose at n=200: BEATS skynet by +0.18 recall and +0.16
        # MRR — the v7 codex audit upgraded lean from "synthetic-tuned" to
        # "the recommended preset for both data types at meaningful sample
        # sizes." (v6 had flagged it as over-aggressive on real prose
        # using only n=50; the n=200 follow-up flipped that finding.)
        # `recency` survives lean — the surface-trap for fresh memories is
        # exactly what lean's loss of `temporal` would otherwise expose.
        if self._retrieval_mode == "lean":
            self._channel_weights.update({
                "bm25": 0.0,
                "temporal": 0.0,
                "salience": 0.0,
            })
        # Trim preset: drop ONLY salience. Conservative middle-ground for
        # callers who want to keep bm25/temporal active (some niche
        # data shapes — e.g. very short corpora — may benefit). On the
        # 2026-04-28 real-prose benchmark trim hit R@5=0.51 vs skynet's
        # 0.42, but lean's 0.60 was the bigger win.
        elif self._retrieval_mode == "trim":
            self._channel_weights.update({
                "salience": 0.0,
            })
        # ColBERT late-interaction channel default weight per preset:
        # skynet=1.2 (the precision lever), advanced/hybrid=0.5,
        # semantic/lean/trim=0. Default-off for the base preset to
        # preserve existing benchmark numbers when the column is empty.
        # Caller-supplied channel_weights (below)
        # override this. Kept OUT of the default dict above so users
        # explicitly opt in via env (MM_COLBERT_ENABLED) or per-recall
        # kwargs.
        if "colbert" not in self._channel_weights:
            if self._retrieval_mode == "skynet":
                self._channel_weights["colbert"] = 1.2
            elif self._retrieval_mode in ("advanced", "hybrid"):
                self._channel_weights["colbert"] = 0.5
            else:
                self._channel_weights["colbert"] = 0.0
        # DAE channel weight — defaults off (0.0) on every preset until
        # the bench validates a recall lift over ColBERT@1.5.  Opt-in
        # via env (MM_DAE_ENABLED=1) or per-recall kwargs.  Pro-gated
        # downstream the same way ColBERT is.
        if "dae" not in self._channel_weights:
            self._channel_weights["dae"] = 0.0
        if isinstance(channel_weights, dict):
            self._channel_weights.update({k: float(v) for k, v in channel_weights.items() if v is not None})

        # ColBERT global toggle:
        #   - MM_COLBERT_ENABLED=1 → write top-32 token blob on every
        #     remember(), and use the colbert channel in recall().
        #   - MM_COLBERT_ENABLED=0 (default) → skip the blob write
        #     entirely. Recall still uses any existing blobs (so a
        #     migration-then-disable workflow keeps cheap reads),
        #     unless the colbert weight is also 0.
        # The opt-in default exists because populating the blob costs
        # ~64 KB/memory; on a 230k-memory corpus that's 14.7 GB extra
        # disk — kept opt-in so the cheap-recall path stays small.
        # ColBERT is a Pro feature.  The env knob still exists (lets a
        # Pro operator toggle the channel without re-issuing the JWT)
        # but it now AND-s with has_feature("colbert"), so a community
        # install with MM_COLBERT_ENABLED=1 quietly stays on the
        # hybrid path — published R@5 = 0.96, no ColBERT bonus.
        _cb_env = (os.environ.get("MM_COLBERT_ENABLED", "0").strip().lower()
                   in ("1", "true", "yes", "on"))
        self._colbert_write_enabled = _cb_env and has_feature("colbert")
        if _cb_env and not self._colbert_write_enabled:
            logger.warning(
                "MM_COLBERT_ENABLED=1 ignored — ColBERT@1.5 is a Pro "
                "feature.  Engine staying on hybrid recall (R@5 = 0.96). "
                "See https://mazemaker.online/#pricing"
            )

        self._hnsw_enabled = use_hnsw
        if self._hnsw_enabled is None:
            self._hnsw_enabled = "auto"
        self._hnsw_index = None
        self._hnsw_ids: list[int] = []
        # Highest memory id already inserted into the live HNSW index. If the
        # current store top-id matches, a recall can skip rebuild entirely.
        # If only larger ids exist (i.e., new appends), we add_items the diff
        # instead of rebuilding from scratch.
        self._hnsw_known: set[int] = set()
        self._hnsw_capacity: int = 0
        self._hnsw_dirty = True
        # Cross-process drift detection. When a separate process (mcp_local,
        # SaaS server, sync_bridge, manual sqlite3 INSERT, etc.) writes to
        # the same SQLite, this process's HNSW + graph cache go silently
        # stale. Periodically peek at SELECT MAX(id) — if it exceeds the
        # max id we've personally indexed, mark the index dirty so the next
        # _ensure_hnsw rebuilds. Cheap (indexed b-tree max), throttled.
        self._hnsw_max_known_id: int = 0
        self._last_drift_check: float = 0.0
        self._drift_check_interval: float = 30.0  # seconds

        self._cpp = None
        self._cpp_id_map: dict[int, int] = {}
        if use_cpp:
            try:
                from cpp_bridge import MazemakerCpp
                self._cpp = MazemakerCpp()
                self._cpp.initialize(dim=self.dim)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("C++ bridge unavailable, falling back to Python: %s", e)
                self._cpp = None

        # GPU recall engine — three-tier resolution:
        #   (1) REMOTE: client-only mode (EMBED_CLIENT_ONLY=1) AND the shared
        #       embed-server has a recall engine attached. Saves ~3 GB VRAM
        #       per client pod by routing recall through the same UNIX socket
        #       as embed. The single canonical GpuRecallEngine lives on the
        #       host (dream_worker) and serves all pods.
        #   (2) LOCAL: cache present at ~/.mazemaker/engine/gpu_cache → load
        #   (3) AUTO-BUILD: cache absent → build from db, then load
        # Anything that fails logs loudly so silent CPU fallbacks are visible.
        self._gpu = None
        if db_path:
            import logging
            _glog = logging.getLogger(__name__)

            # Tier 1: remote recall via shared embed socket
            tried_remote = False
            if os.environ.get("EMBED_CLIENT_ONLY"):
                try:
                    backend = getattr(self.embedder, "backend", None)
                    sclient = getattr(backend, "_shared_client", None) or getattr(backend, "_client", None)
                    if sclient is not None and hasattr(sclient, "recall_status"):
                        from embed_provider import RemoteGpuRecallClient
                        remote = RemoteGpuRecallClient(sclient)
                        tried_remote = True
                        if remote.probe():
                            self._gpu = remote
                            _glog.info(
                                "GPU recall ARMED (remote, via shared embed socket): %d vectors on canonical server",
                                remote._n,
                            )
                except Exception as exc_r:
                    _glog.warning("GPU recall remote-init failed (will try local): %s", exc_r)

            # Tier 2/3: local engine — only if remote didn't take. In strict
            # client-only mode, refuse to load a local engine even on remote
            # failure: the operator's contract is "one canonical engine, no
            # redundant pod copies".
            if self._gpu is None and not tried_remote:
                try:
                    from gpu_recall import GpuRecallEngine
                    # Per-DB cache: GpuRecallEngine derives an isolated cache subdir
                    # from db_path. Backwards compat: production DBs still use the
                    # global cache dir. Eliminates cross-DB contamination when this
                    # process opens a benchmark / test / per-conv DB.
                    eng = GpuRecallEngine(db_path=db_path)
                    if eng.load(embed_fn=self.embedder.embed, embed_batch_fn=getattr(self.embedder, "embed_batch", None)):
                        # Fingerprint check: cache row count must match the DB's
                        # current memory count. Mismatch indicates either a stale
                        # cache from a prior version or accidental sharing — force
                        # rebuild rather than silently serve wrong embeddings.
                        cached_n = eng._emb_tensor.shape[0] if eng._emb_tensor is not None else 0
                        # SQLiteStore exposes the connection as `.conn`. The
                        # previous `.connection()` call raised AttributeError
                        # every time and was silently swallowed below, so the
                        # entire fingerprint check was a permanent no-op —
                        # stale caches always loaded "accept cache".
                        store_conn = getattr(self.store, "conn", None)
                        if store_conn is None:
                            db_n = cached_n  # non-SQLite backend, skip check
                        else:
                            try:
                                db_n = store_conn.execute(
                                    "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
                                ).fetchone()[0]
                            except sqlite3.Error as exc_fp:
                                _glog.warning(
                                    "GPU cache fingerprint check failed (%s) — accepting cache",
                                    exc_fp,
                                )
                                db_n = cached_n
                        if cached_n != db_n:
                            _glog.warning(
                                "GPU recall cache STALE (cached=%d, db=%d) — rebuilding",
                                cached_n, db_n,
                            )
                            try:
                                from build_gpu_cache import build  # type: ignore[import]
                                from pathlib import Path as _P
                                build(_P(db_path), eng.cache_dir)
                                eng = GpuRecallEngine(db_path=db_path)
                                if eng.load(embed_fn=self.embedder.embed, embed_batch_fn=getattr(self.embedder, "embed_batch", None)):
                                    self._gpu = eng
                                    _glog.info(
                                        "GPU recall ARMED post-rebuild: %d vectors on %s",
                                        eng._emb_tensor.shape[0] if eng._emb_tensor is not None else 0,
                                        eng._device,
                                    )
                            except Exception as rebuild_exc:
                                _glog.warning("GPU recall cache rebuild failed: %s", rebuild_exc)
                        else:
                            self._gpu = eng
                            _glog.info(
                                "GPU recall ARMED: %d vectors on %s (cache dir: %s)",
                                cached_n, eng._device, eng.cache_dir,
                            )
                    else:
                        # Detect PG-backend: the SQLite gpu_cache+auto-build
                        # path is structurally inapplicable (db_path is
                        # /dev/null or any non-SQLite sentinel). Go straight
                        # to load_from_store — works for any store exposing
                        # get_all(). Stays silent unless something fails.
                        is_pg_backend = (
                            (os.environ.get("MM_DB_BACKEND") or "").strip().lower() == "postgres"
                            or str(db_path) in ("/dev/null", "")
                        )
                        built_from_cache = False
                        if not is_pg_backend:
                            _glog.warning(
                                "GPU recall cache absent — auto-building from %s", db_path,
                            )
                            try:
                                from build_gpu_cache import build  # type: ignore[import]
                                from pathlib import Path as _P
                                build(_P(db_path), eng.cache_dir)
                                if eng.load(embed_fn=self.embedder.embed, embed_batch_fn=getattr(self.embedder, "embed_batch", None)):
                                    self._gpu = eng
                                    built_from_cache = True
                                    _glog.info(
                                        "GPU recall ARMED post-build: %d vectors (cache dir: %s)",
                                        eng._emb_tensor.shape[0] if eng._emb_tensor is not None else 0,
                                        eng.cache_dir,
                                    )
                            except BaseException as exc_b:
                                _glog.warning(
                                    "GPU recall auto-build skipped (cache-build path): %s",
                                    exc_b,
                                )
                        # Backend-direct path. Pulls embeddings straight from
                        # whatever store is live and uploads them to the GPU
                        # tensor — works for PG, MSSQL, fresh SQLite, anything
                        # that exposes store.get_all(). Logged at WARNING level
                        # so it's visible even when INFO is filtered.
                        if not built_from_cache and self._gpu is None:
                            try:
                                if eng.load_from_store(
                                    self.store,
                                    embed_fn=self.embedder.embed,
                                    embed_batch_fn=getattr(self.embedder, "embed_batch", None),
                                ):
                                    self._gpu = eng
                                    _glog.warning(
                                        "GPU recall ARMED (load_from_store): %d vectors on %s",
                                        eng._emb_tensor.shape[0] if eng._emb_tensor is not None else 0,
                                        eng._device,
                                    )
                                else:
                                    _glog.warning(
                                        "GPU recall: load_from_store returned False — recall will run on CPU/numpy"
                                    )
                            except BaseException as exc_lfs:
                                _glog.warning(
                                    "GPU recall load_from_store failed: %s — CPU/numpy fallback",
                                    exc_lfs,
                                )
                except BaseException as exc:
                    _glog.warning(
                        "GPU recall init skipped (recall will run on CPU/numpy): %s", exc
                    )
            elif self._gpu is None and tried_remote:
                _glog.warning(
                    "GPU recall remote unavailable + EMBED_CLIENT_ONLY=1 — recall on CPU/numpy. "
                    "Refusing to ARM a redundant local engine (operator policy: one canonical engine)."
                )

        self._graph_nodes: dict[int, dict] = {}
        # Counter set by _load_from_store; surfaced via stats() so operators
        # can spot a corrupt-on-arrival DB (mixed-dim memories from a prior
        # backend swap) without grepping logs.
        self._quarantined_dim: int = 0
        if not self._lazy_graph:
            self._load_from_store()

    # -- one-model invariant -------------------------------------------------

    def _compute_embed_fingerprint(self) -> str:
        """Identity string for the active embedding backend.

        Combines class name, declared dim, and (when available) the model
        name. Stable across processes for the same backend selection.
        """
        backend = getattr(self.embedder, "backend", self.embedder)
        cls = type(backend).__name__
        model = getattr(backend, "model_name", None) or getattr(backend, "name", "")
        return f"{cls}::{self.dim}::{model}"

    def _pin_fingerprint_if_unset(self) -> None:
        """Write the active backend's fingerprint to db_meta if absent.

        Called from remember() before the first write so the pin only
        happens when the DB actually gets a memory. Read-only sessions
        that never call remember() leave the fingerprint unset and don't
        lock the user into whichever backend they happened to use for
        exploration.
        """
        if not hasattr(self.store, "get_meta"):
            return
        if self.store.get_meta("embed_fingerprint") is None:
            self.store.set_meta("embed_fingerprint", self._embed_fingerprint)
        if self.store.get_meta("embed_dim") is None:
            self.store.set_meta("embed_dim", str(self.dim))

    def _enforce_dim_lock(self) -> tuple[bool, str]:
        """Compare the active backend against the DB's stored fingerprint.

        Read-only check: this method NEVER writes to db_meta. The
        fingerprint is pinned on first remember() (see
        _pin_fingerprint_if_unset). Empty DBs and mid-init reads are
        therefore harmless; only an actual mismatch against a
        previously-pinned fingerprint produces dim_locked=False.

        Stores without db_meta support get a \"locked-on-current-backend\"
        result so writes proceed; the one-model invariant on those stores
        is enforced server-side via the schema's vector_dim column.
        """
        if not hasattr(self.store, "get_meta"):
            return True, ""
        stored = self.store.get_meta("embed_fingerprint")
        if stored is None:
            # Empty DB or read-only init — no pin yet. The first remember()
            # call will pin via _pin_fingerprint_if_unset(). Treat as
            # locked-on-current-backend so that initial writes succeed.
            return True, ""
        try:
            stored_dim = int(self.store.get_meta("embed_dim") or "0")
        except ValueError:
            stored_dim = 0
        if stored_dim and stored_dim != self.dim:
            reason = (
                f"DB was pinned at dim={stored_dim} ({stored}); "
                f"current backend is dim={self.dim} ({self._embed_fingerprint}). "
                f"New writes will be rejected; old memories quarantined from "
                f"similarity until re-embedded."
            )
            import logging
            logging.getLogger(__name__).warning("[neural] dim mismatch — %s", reason)
            return False, reason
        # Same dim, possibly different model. Update the fingerprint so the
        # next open gets the precise identity, but keep the lock active.
        if stored != self._embed_fingerprint:
            self.store.set_meta("embed_fingerprint", self._embed_fingerprint)
        return True, ""

    # -- graph/index hydration ------------------------------------------------

    def _load_from_store(self):
        all_mems = self.store.get_all()
        quarantined = 0
        for mem in all_mems:
            emb = mem.get("embedding") or []
            # Quarantine: don't load embeddings whose dim disagrees with the
            # active backend. Keeping the row in DB preserves the user's
            # data; excluding it from _graph_nodes prevents zip-truncation
            # (already guarded by iter 11) and keeps hot-path code simple.
            if emb and len(emb) != self.dim:
                quarantined += 1
                emb = []
            self._graph_nodes[mem["id"]] = {
                "embedding": emb,
                "label": mem["label"],
                "content": mem.get("content", ""),
                "connections": {},
            }
        self._quarantined_dim = quarantined

        # Bulk-load all edges in one SQL pass and bucket them onto each node,
        # rather than firing one get_connections query per memory. With 10K
        # memories the per-row variant burned 10K SQL round-trips on startup;
        # the bulk variant is one round-trip plus a Python loop.
        if hasattr(self.store, "get_all_connections"):
            for edge in self.store.get_all_connections():
                src = int(edge["source"])
                tgt = int(edge["target"])
                w = float(edge["weight"])
                # _graph_nodes already exists for known memories; the
                # default-dict-style branch handles edges pointing to memories
                # that aren't in _graph_nodes (orphan edges that pre-date a
                # delete) without crashing.
                node_s = self._graph_nodes.setdefault(src, {
                    "embedding": [], "label": f"memory:{src}", "content": "", "connections": {},
                })
                node_t = self._graph_nodes.setdefault(tgt, {
                    "embedding": [], "label": f"memory:{tgt}", "content": "", "connections": {},
                })
                node_s.setdefault("connections", {})[tgt] = w
                node_t.setdefault("connections", {})[src] = w
        else:
            # Non-SQLite store path (no get_all_connections) — fall back to
            # the per-memory query so behaviour is preserved.
            for mem in all_mems:
                self._refresh_connections(mem["id"])

        if self._cpp:
            for mem in all_mems:
                emb = mem.get("embedding", [])
                if emb and len(emb) == self.dim:
                    try:
                        cpp_id = self._cpp.store(emb, mem.get("label", ""), mem.get("content", ""))
                        self._cpp_id_map[cpp_id] = mem["id"]
                    except Exception:
                        pass

    def _refresh_connections(self, mem_id: int, at_time: Optional[float] = None) -> None:
        if mem_id not in self._graph_nodes:
            return
        self._graph_nodes[mem_id]["connections"] = {}
        for c in self.store.get_connections(mem_id, at_time=at_time):
            other = c["target"] if c["source"] == mem_id else c["source"]
            self._graph_nodes[mem_id]["connections"][other] = float(c["weight"] or 0.0)
            if other not in self._graph_nodes:
                self._graph_nodes[other] = {"embedding": [], "label": f"memory:{other}", "content": "", "connections": {}}

    def _ensure_node(self, mem_id: int, at_time: Optional[float] = None) -> bool:
        """Make sure a graph node exists in the cache; populate from store on miss.

        Two-tier optimisation:

          1. Cache-resident, non-placeholder, non-time-travel hit: skip the
             metadata lookup (store.get) but STILL refresh connections, so
             external mutations via store.add_connection are visible.
          2. Cache miss / placeholder / time-travel: full populate path.

        Iter 22 originally returned True from the cache-hit branch without
        calling _refresh_connections, which broke any caller that mutated the
        DB directly (e.g. test_ppr_think_engine seeded edges via
        store.add_connection then expected the in-memory graph to see them).
        We keep the metadata-lookup savings — that was the iter 22 win — but
        no longer skip the connection refresh.
        """
        cached = self._graph_nodes.get(mem_id)
        # Defensive None-coercion: dict.get("label", "") returns None when the
        # key exists with a None value (SQLite NULL columns), not the default.
        # Hermes hit "'NoneType' object has no attribute 'startswith'" here in
        # production. Coerce to empty string before calling startswith.
        cached_label = (cached.get("label") if cached else None) or ""
        cache_hit = (
            at_time is None
            and cached is not None
            and not cached_label.startswith("memory:")
        )
        if not cache_hit:
            mem = self.store.get(mem_id)
            if not mem:
                return False
            self._graph_nodes[mem_id] = {
                "embedding": mem.get("embedding", []),
                "label": mem.get("label", ""),
                "content": mem.get("content", ""),
                "connections": (cached or {}).get("connections", {}),
            }
        # Refresh connections in BOTH branches. On the cache-hit branch this
        # is the only SQL query (was 2 before iter 22, now 1, was 0 in the
        # buggy iter 22 fast-path); on the miss branch it's part of the
        # original two-query populate.
        self._refresh_connections(mem_id, at_time=at_time)
        return True

    def _check_external_drift(self) -> None:
        """Detect rows added to SQLite by another process.

        Throttled to once every `self._drift_check_interval` seconds (default
        30s). When the SQLite top-id exceeds the highest id this process has
        seen, mark `_hnsw_dirty=True` so the next `_ensure_hnsw` picks up the
        new rows. Without this, separate processes (mcp_local, SaaS server,
        sync_bridge, manual sqlite3 INSERT) can write rows that the running
        recall pipeline never sees — they're in SQLite but not in HNSW or
        the in-memory graph cache.

        Costs: one indexed `MAX(id)` query every 30s. Microseconds.
        Trades latency for staleness — at the default interval, drift is
        visible within ~30s of the external write.
        """
        now = time.time()
        if now - self._last_drift_check < self._drift_check_interval:
            return
        self._last_drift_check = now
        try:
            live_top = self.store.get_max_id()
        except Exception:
            return
        if live_top > self._hnsw_max_known_id:
            # External drift detected. Mark stale; next recall rebuilds.
            self._hnsw_dirty = True
            # Also clear the graph-node cache for any new ids — they'll be
            # lazily re-hydrated by `_ensure_node` on first access.
            # (Don't blow away ALL nodes; only the ones we definitely missed.)
            self._hnsw_max_known_id = live_top

    def _ensure_hnsw(self) -> bool:
        """Build or incrementally update the HNSW ANN index.

        First call constructs from scratch (paying O(N log N) once).
        Subsequent calls after writes only add the new memory ids via
        hnswlib.add_items, which is O(M log N) for M new items — way
        cheaper than the previous always-rebuild path that paid O(N log N)
        on every recall after even a single write.

        Falls back to full rebuild only when the live capacity is exhausted
        or when the index contains ids the store no longer has (deletes).

        Now also detects cross-process drift via `_check_external_drift`
        so concurrent writers (mcp_local, SaaS, sync_bridge) can share a
        SQLite without each running process becoming silently stale.
        """
        if self._hnsw_enabled is False or str(self._hnsw_enabled).lower() in {"0", "false", "no", "off"}:
            return False
        # Cheap pre-check: if another process appended rows since our last
        # build, mark the index dirty so the cached fast-path below misses.
        self._check_external_drift()
        if self._hnsw_index is not None and not self._hnsw_dirty:
            return True
        try:
            import hnswlib
            import numpy as np
        except Exception:
            return False
        mems = self.store.get_all()
        if not mems:
            return False
        if str(self._hnsw_enabled).lower() == "auto" and len(mems) < 1000:
            return False

        # Filter to dim-matching embeddings (quarantine awareness)
        usable = [(int(m["id"]), m["embedding"]) for m in mems
                  if (m.get("embedding") and len(m["embedding"]) == self.dim)]
        if not usable:
            return False
        store_ids = {mid for mid, _ in usable}

        # Decide: incremental add or full rebuild?
        new_ids = store_ids - self._hnsw_known
        deleted_ids = self._hnsw_known - store_ids
        can_increment = (
            self._hnsw_index is not None
            and not deleted_ids
            and (len(self._hnsw_known) + len(new_ids)) <= self._hnsw_capacity
        )

        if can_increment and new_ids:
            new_pairs = [(mid, emb) for mid, emb in usable if mid in new_ids]
            ids_arr = np.asarray([mid for mid, _ in new_pairs], dtype=np.int64)
            vecs_arr = np.asarray([emb for _, emb in new_pairs], dtype=np.float32)
            self._hnsw_index.add_items(vecs_arr, ids_arr)
            self._hnsw_known.update(new_ids)
            self._hnsw_ids = list(self._hnsw_known)
            self._hnsw_dirty = False
            # Update drift baseline so the next external-drift check uses
            # the current top-id, not the pre-incremental value.
            if new_ids:
                self._hnsw_max_known_id = max(self._hnsw_max_known_id, max(new_ids))
            return True

        # Full rebuild (cold start, deletions, or capacity overflow)
        ids = [mid for mid, _ in usable]
        vecs = [emb for _, emb in usable]
        capacity = max(len(ids) * 2, len(ids) + 1024)
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=capacity, ef_construction=200, M=16)
        index.add_items(np.asarray(vecs, dtype=np.float32), np.asarray(ids, dtype=np.int64))
        index.set_ef(64)
        self._hnsw_index = index
        self._hnsw_ids = ids
        self._hnsw_known = set(ids)
        self._hnsw_capacity = capacity
        self._hnsw_dirty = False
        # Refresh drift-detection baseline after a full rebuild.
        self._hnsw_max_known_id = max(ids) if ids else 0
        return True

    # -- write path -----------------------------------------------------------

    def remember(self, text: str, label: str = "", detect_conflicts: bool = True, auto_connect: bool = True, detect_supersedes: bool = True) -> int:
        embedding = self.embedder.embed(text)
        # Hard guard: refuse to write a vector whose dim disagrees with the
        # DB's pinned dim. The alternative (writing it anyway) corrupts every
        # subsequent recall against this row.
        if not self._dim_locked:
            raise RuntimeError(
                f"refusing to write: {self._dim_mismatch_reason} "
                "Re-open with the original embedding backend, or drop the DB."
            )
        if len(embedding) != self.dim:
            raise RuntimeError(
                f"embedder produced dim={len(embedding)}, expected {self.dim} "
                f"({self._embed_fingerprint})"
            )
        # First write to a fresh DB pins the embedding fingerprint into
        # db_meta. Read-only sessions that never call remember() leave the
        # DB un-pinned, so they don't lock a different-backend operator out
        # of the same DB later.
        self._pin_fingerprint_if_unset()
        label = label or text[:60]

        if detect_conflicts and label:
            for other in self.store.find_by_label(label):
                old_content = other.get("content", "") or ""
                if old_content.strip() == text.strip():
                    return int(other["id"])
                # Same label is not enough — require high embedding similarity
                # before treating this as a conflict to merge.  Without this,
                # common labels like "bug"/"decision" turn every write into a
                # destructive overwrite of the first existing memory.
                # NOTE: find_by_label already SELECTs the embedding column, so
                # other["embedding"] is populated (see MemoryStore.find_by_label).
                other_emb = other.get("embedding") or []
                # Fail closed: missing or dim-mismatched neighbour embedding
                # means we cannot verify similarity, so we MUST NOT fuse —
                # the previous `if other_emb and cos(...) < 0.85: continue`
                # short-circuited on empty embeddings and let the destructive
                # fusion path run unchecked. _cosine_similarity already
                # returns 0.0 on dim mismatch (iter 11); explicit empty-list
                # guard covers the legacy NULL-blob rows.
                if not other_emb:
                    continue
                if len(other_emb) != self.dim:
                    continue
                if self._cosine_similarity(embedding, other_emb) < 0.85:
                    continue
                if self._content_differs(old_content, text) or old_content.strip() != text.strip():
                    fused = self._fuse_conflict(old_content, text)
                    # Re-embed the fused content. Previously we stored
                    # `fused` against `embedding` (the embedding of the
                    # incoming `text`), so every recall against this memory
                    # scored against text it no longer contains. Drift
                    # compounded across repeated fusions. Falls back to the
                    # incoming embedding only if re-embed raises — the
                    # original behaviour, but logged.
                    try:
                        fused_embedding = self.embedder.embed(fused)
                        if len(fused_embedding) != self.dim:
                            raise RuntimeError(
                                f"re-embed dim mismatch ({len(fused_embedding)} != {self.dim})"
                            )
                    except Exception as exc_re:
                        logger.warning(
                            "conflict-fuse re-embed failed (%s) — keeping incoming embedding",
                            exc_re,
                        )
                        fused_embedding = embedding
                    self.store.add_revision(int(other["id"]), old_content, text, "conflict_fusion")
                    self.store.update_memory(int(other["id"]), fused, fused_embedding, label=label)
                    # ColBERT tokens were extracted from the OLD content
                    # — re-encode for the fused text so the rerank channel
                    # stays consistent. Same opt-in gate as the fresh-write
                    # path. Failure is non-fatal.
                    if self._colbert_write_enabled:
                        try:
                            from colbert_helper import encode_tokens, pack_tokens
                            arr = encode_tokens(fused)
                            if arr is not None:
                                self.store.set_colbert_tokens(int(other["id"]), pack_tokens(arr))
                        except Exception:
                            logger.debug("colbert: refresh on conflict-fuse failed", exc_info=True)
                    # Cache MUST mirror what was actually written to DB.
                    # Was setting content=text here while DB held `fused`,
                    # so recall via _graph_nodes saw the raw new content but
                    # recall via store.get_many saw the [CANONICAL]/[PREVIOUSLY]
                    # marker — same memory, two contents, depending on path.
                    self._graph_nodes[int(other["id"])] = {
                        "embedding": fused_embedding, "label": label,
                        "content": fused, "connections": {},
                    }
                    self._refresh_connections(int(other["id"]))
                    self._hnsw_dirty = True
                    return int(other["id"])

        mem_id = self.store.store(label, text, embedding)
        self._graph_nodes[mem_id] = {"embedding": embedding, "label": label, "content": text, "connections": {}}

        # ColBERT token cache. Opt-in via MM_COLBERT_ENABLED. Failure here
        # is non-fatal — the dense embedding is the source of truth; the
        # colbert blob is purely a precision-lever rerank channel. Logged
        # at debug so a misconfigured environment doesn't spam the log.
        if self._colbert_write_enabled:
            try:
                from colbert_helper import encode_tokens, pack_tokens
                arr = encode_tokens(text)
                if arr is not None:
                    self.store.set_colbert_tokens(mem_id, pack_tokens(arr))
            except Exception:
                logger.debug("colbert: token-cache write failed for mem_id=%s", mem_id, exc_info=True)

        if self._cpp:
            try:
                cpp_id = self._cpp.store(embedding, label, text)
                self._cpp_id_map[cpp_id] = mem_id
            except Exception:
                pass
        self._hnsw_dirty = True

        # GPU recall hot-path append — keeps the in-GPU tensor in sync with
        # the SQLite write so new memories are GPU-searchable immediately
        # (no manual cache rebuild). Silent no-op if _gpu is unloaded.
        if self._gpu is not None:
            try:
                self._gpu.add_one(mem_id, label, text, embedding)
            except Exception:
                # Don't break a remember() call if the GPU side glitches —
                # the SQLite write already succeeded; the cache will heal
                # on next process restart (auto-build).
                import logging
                logging.getLogger(__name__).warning(
                    "GPU recall add_one failed for mem_id=%s; cache now stale "
                    "until next rebuild", mem_id, exc_info=True,
                )

        if auto_connect:
            self._auto_connect(mem_id, embedding, text)

        # Task 3 — ingest-time SUPERSEDES detection.
        # After the fresh memory is stored, scan recent memories for
        # high-similarity pairs with *different* numeric tokens.  This
        # unifies the ingest-time and dream-time supersession paths so
        # downstream consumers only need to read the connections table.
        # Intentionally non-blocking: any exception is logged and ignored
        # so a DB hiccup can't break a remember() call.
        if detect_supersedes:
            self._detect_supersedes_at_ingest(mem_id, embedding, text)

        return mem_id

    def remember_batch(
        self,
        rows: "list[dict]",
        *,
        detect_conflicts: bool = False,
        auto_connect: bool = False,
        detect_supersedes: bool = True,
    ) -> "list[int]":
        if not rows:
            return []
        if detect_conflicts or auto_connect:
            # Slow-path: per-row remember() preserves conflict-fusion +
            # auto_connect side-effects. Bulk ingest hot paths leave both
            # off, so this is for callers that explicitly opted in.
            logger.warning(
                "remember_batch: detect_conflicts/auto_connect requested → "
                "falling back to per-row remember() (slow path)"
            )
            out: list[int] = []
            for r in rows:
                out.append(self.remember(
                    r.get("text", ""),
                    label=r.get("label") or "",
                    detect_conflicts=detect_conflicts,
                    auto_connect=auto_connect,
                    detect_supersedes=detect_supersedes,
                ))
            return out

        if not self._dim_locked:
            raise RuntimeError(
                f"refusing to write: {self._dim_mismatch_reason} "
                "Re-open with the original embedding backend, or drop the DB."
            )

        texts = [r.get("text", "") for r in rows]
        embed_batch_fn = getattr(self.embedder, "embed_batch", None)
        if embed_batch_fn is not None:
            embeddings = embed_batch_fn(texts)
        else:
            embeddings = [self.embedder.embed(t) for t in texts]

        if len(embeddings) != len(rows):
            raise RuntimeError(
                f"embed_batch returned {len(embeddings)} vectors for "
                f"{len(rows)} texts"
            )
        for emb in embeddings:
            if len(emb) != self.dim:
                raise RuntimeError(
                    f"embedder produced dim={len(emb)}, expected {self.dim} "
                    f"({self._embed_fingerprint})"
                )

        self._pin_fingerprint_if_unset()

        store_rows: list[dict] = []
        for r, emb in zip(rows, embeddings):
            text = r.get("text", "")
            label = r.get("label") or text[:60]
            store_rows.append({
                "label": label,
                "content": text,
                "embedding": emb,
                "salience": (r.get("metadata") or {}).get("salience"),
            })

        ids = self.store.remember_batch(store_rows)

        for mem_id, sr in zip(ids, store_rows):
            self._graph_nodes[int(mem_id)] = {
                "embedding": sr["embedding"],
                "label": sr["label"],
                "content": sr["content"],
                "connections": {},
            }
        self._hnsw_dirty = True

        if self._gpu is not None:
            for mem_id, sr in zip(ids, store_rows):
                try:
                    self._gpu.add_one(int(mem_id), sr["label"], sr["content"], sr["embedding"])
                except Exception:
                    logger.warning(
                        "GPU recall add_one failed for mem_id=%s; cache now stale "
                        "until next rebuild", mem_id, exc_info=True,
                    )
                    break

        return [int(i) for i in ids]

    # -- numeric token extraction shared by ingest + dream paths ----------

    _SUPERSEDES_NUMERIC_RE = re.compile(
        r"""
        \$?\d[\d,]*(?:\.\d+)?[KkMmBb]?   # dollar amounts or any number
        (?:\s*(?:GB|MB|KB|TB|GHz|MHz|kHz|Hz|kg|km|cm|mm|inches|ft|inch|lbs|lb|hrs?|mins?))?
        \b
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    _SUPERSEDES_SIM_THRESHOLD: float = 0.85
    _SUPERSEDES_SCAN_WINDOW: int = 2000  # only scan this many most-recent nodes

    def _extract_numeric_tokens(self, text: str) -> frozenset:
        """Return normalised numeric tokens from text (strip commas)."""
        # Fast-path: no digit character at all → skip the full regex.
        if not re.search(r"\d", text):
            return frozenset()
        raw = self._SUPERSEDES_NUMERIC_RE.findall(text)
        return frozenset(t.replace(",", "").strip() for t in raw)

    def _detect_supersedes_at_ingest(
        self, new_id: int, new_emb: list[float], new_text: str
    ) -> None:
        """Write SUPERSEDES edges from older memories to new_id.

        Embedding-similarity pathway: cosine >= _SUPERSEDES_SIM_THRESHOLD,
        gated on differing numeric tokens (real update, not a dup) and
        other memory predating new_id.

        Writes a directed supersedes edge: source=older_id, target=new_id.
        Idempotent — add_connection skips existing edges via UPDATE.
        """
        new_tokens = self._extract_numeric_tokens(new_text)
        if not new_tokens:
            return  # no numeric content → skip (nothing to supersede)

        threshold = self._SUPERSEDES_SIM_THRESHOLD
        new_norm = sum(v * v for v in new_emb) ** 0.5
        if new_norm == 0.0:
            return

        # Only scan the most-recent N nodes — supersedes is temporally local
        # (an updated fact about the same entity).  W/o this window the loop
        # is O(N²) in the graph size, which kills bulk-ingest throughput.
        window_nodes = list(self._graph_nodes.items())[-self._SUPERSEDES_SCAN_WINDOW:]

        for other_id, node in window_nodes:
            other_id = int(other_id)
            if other_id == new_id:
                continue
            if other_id >= new_id:
                continue
            other_emb = node.get("embedding") or []
            if not other_emb or len(other_emb) != len(new_emb):
                continue
            other_text = node.get("content") or ""
            other_tokens = self._extract_numeric_tokens(other_text)
            if not other_tokens or other_tokens == new_tokens:
                continue
            dot = sum(a * b for a, b in zip(new_emb, other_emb))
            other_norm = sum(v * v for v in other_emb) ** 0.5
            denom = new_norm * other_norm
            sim = dot / denom if denom else 0.0
            if sim < threshold:
                continue
            try:
                self.store.add_connection(
                    other_id, new_id, sim, edge_type="supersedes"
                )
            except Exception:
                pass

    def _auto_connect(self, mem_id: int, embedding: list[float], text: str) -> None:
        """Connect a freshly-stored memory to its semantic neighbours.

        Two strategies depending on graph size:

          1. HNSW ANN fast-path (used when _ensure_hnsw succeeds — typically
             >= 1000 memories). Pulls the top ~32 candidates above the 0.70
             cosine floor in O(log N) and only computes exact cosine on
             those. The brute-force loop is O(N), which on a 100K-memory
             store added measurable latency to every remember() call.

          2. Brute-force fallback (small graphs, HNSW disabled, or hnswlib
             missing). Identical to the previous implementation: walk
             _graph_nodes, exact-cosine each, threshold at 0.70.
        """
        candidates: list[tuple[int, float]] = []  # (other_id, similarity)

        if self._ensure_hnsw():
            try:
                import numpy as np
                # Fetch a generous neighbourhood; HNSW recall@k is high but
                # not perfect, so over-fetching ensures the 0.70+ true
                # positives aren't missed. Capped at the index size to avoid
                # asking hnswlib for more than it knows.
                k_probe = min(64, len(self._hnsw_ids))
                if k_probe > 0:
                    labels, distances = self._hnsw_index.knn_query(
                        np.asarray([embedding], dtype=np.float32),
                        k=k_probe,
                    )
                    for nid, dist in zip(labels[0], distances[0]):
                        nid = int(nid)
                        if nid == mem_id:
                            continue
                        sim = max(0.0, 1.0 - float(dist))
                        if sim > 0.70:
                            candidates.append((nid, sim))
            except Exception:
                # Fall through to brute-force if hnswlib chokes.
                candidates = []

        if not candidates:
            for other_id, node in list(self._graph_nodes.items()):
                other_id = int(other_id)
                if other_id == mem_id:
                    continue
                other_emb = node.get("embedding") or []
                if not other_emb:
                    continue
                sim = self._cosine_similarity(embedding, other_emb)
                # 0.45 is a noise floor with FastEmbed/e5-large — produces
                # O(n²) edge growth (spurious matches); 0.70 keeps
                # semantically-related pairs while killing low-signal
                # connections.
                if sim > 0.70:
                    candidates.append((other_id, sim))

        # Group candidates by edge_type and bulk-write via
        # add_connections_batch — single COPY+upsert per type instead of
        # K round-trips. In-memory graph mirror updates unchanged.
        by_type: dict[str, list[tuple[int, int, float]]] = {}
        for other_id, sim in candidates:
            other_node = self._graph_nodes.get(other_id, {})
            edge_type = self._infer_edge_type(text, other_node.get("content", ""))
            by_type.setdefault(edge_type, []).append((mem_id, other_id, sim))
        if by_type and hasattr(self.store, "add_connections_batch"):
            for et, pairs in by_type.items():
                try:
                    self.store.add_connections_batch(pairs, edge_type=et)
                except Exception:
                    for s, t, w in pairs:
                        try:
                            self.store.add_connection(s, t, w, edge_type=et)
                        except Exception:
                            pass
        else:
            for other_id, sim in candidates:
                other_node = self._graph_nodes.get(other_id, {})
                edge_type = self._infer_edge_type(text, other_node.get("content", ""))
                self.store.add_connection(mem_id, other_id, sim, edge_type=edge_type)
        for other_id, sim in candidates:
            self._graph_nodes.setdefault(other_id, {
                "embedding": [], "label": f"memory:{other_id}",
                "content": "", "connections": {},
            }).setdefault("connections", {})[mem_id] = sim
            self._graph_nodes[mem_id].setdefault("connections", {})[other_id] = sim

    def _fuse_conflict(self, old_text: str, new_text: str) -> str:
        old_clean = old_text.replace("[SUPERSEDED]", "").replace("[UPDATED TO]", "").replace("[CANONICAL]", "").replace("[PREVIOUSLY]", "").strip()
        if len(old_clean) > 1200:
            old_clean = old_clean[:1200] + " ...[truncated]"
        return f"[CANONICAL] {new_text.strip()}\n[PREVIOUSLY] {old_clean}"

    def _find_conflicts(self, new_text: str, new_embedding: list[float], threshold: float = 0.6) -> dict:
        conflicts = {}
        for mem in self.store.get_all():
            content = mem.get("content", "") or ""
            if content.startswith("[SUPERSEDED]"):
                continue
            sim = self._cosine_similarity(new_embedding, mem["embedding"])
            if sim > threshold:
                conflicts[mem["id"]] = {"similarity": sim, "content": content, "label": mem.get("label", "")}
        return conflicts

    def _content_differs(self, old_text: str, new_text: str) -> bool:
        old_clean = old_text.replace("[SUPERSEDED]", "").replace("[UPDATED TO]", "").replace("[CANONICAL]", "").replace("[PREVIOUSLY]", "").strip()
        old_nums = set(re.findall(r"\d+\.?\d*", old_clean))
        new_nums = set(re.findall(r"\d+\.?\d*", new_text))
        if old_nums != new_nums and old_nums and new_nums:
            return True
        negations = {"not", "n't", "never", "no", "none", "nothing", "nowhere"}
        if any(n in old_clean.lower().split() for n in negations) != any(n in new_text.lower().split() for n in negations):
            return True
        old_dates = set(re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}", old_clean))
        new_dates = set(re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}", new_text))
        if old_dates != new_dates and old_dates and new_dates:
            return True
        return old_clean.strip() != new_text.strip()

    @staticmethod
    def _infer_edge_type(a: str, b: str) -> str:
        joined = f"{a} {b}".lower()
        if any(w in joined for w in ("because", "caused", "causes", "due to", "therefore", "leads to")):
            return "causal"
        if any(w in joined for w in ("supersedes", "replaces", "updated to", "previously")):
            return "supersedes"
        return "similar"

    # -- scoring helpers ------------------------------------------------------

    def _compute_temporal_score(self, mem_id: int, now: float) -> float:
        try:
            mem = self.store.get(mem_id, include_embedding=False)
            if mem and mem.get("created_at"):
                return self._compute_temporal_score_from_mem(mem, now)
            return 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _compute_temporal_score_from_mem(mem: dict, now: float, half_life_hours: float = 24.0 * 7) -> float:
        created = mem.get("created_at") or now
        try:
            age_hours = max(0.0, (now - float(created)) / 3600.0)
        except Exception:
            return 0.5
        return math.exp(-math.log(2) * age_hours / max(1.0, half_life_hours))

    def _effective_salience(self, mem: dict, now: float) -> float:
        base = float(mem.get("salience", 1.0) or 1.0)
        access = float(mem.get("access_count", 0) or 0)
        created = float(mem.get("created_at", now) or now)
        days = max(0.0, (now - created) / 86400.0)
        factor = base * math.exp(-self._salience_decay_k * days) + math.log1p(access) * 0.05
        return max(0.1, min(2.0, factor))

    # -- retrieval channels ---------------------------------------------------

    def _semantic_candidates(self, query: str, query_vec: list[float], limit: int) -> list[dict]:
        # Store-native pgvector HNSW FIRST when the store knows how. The
        # PG path used to fall through to `get_all()` + Python cosine
        # across the whole corpus — 75 s for a 186 k-row schema. The
        # HNSW index on `embedding` lives in PG already; let it work.
        if hasattr(self.store, "search_semantic"):
            try:
                res = self.store.search_semantic(query_vec, limit=limit)
                if res:
                    return res
            except Exception:
                pass

        # GPU recall path next when armed — the customer paid for the GPU
        # variant of the pod (or the host has CUDA), they expect it to be
        # the path that actually runs. HNSW (CPU) is fast on small DBs but
        # leaves the loaded GPU tensor idle, which is wasteful and breaks
        # the strict-mode promise that device=cuda means cuda.
        if self._gpu:
            try:
                gpu_results = self._gpu.recall(query, k=limit)
                if gpu_results:
                    return [{"id": int(r["id"]), "score": float(r.get("similarity", 0.0)), "similarity": float(r.get("similarity", 0.0)), "channel": "semantic"} for r in gpu_results]
            except Exception:
                pass

        # HNSW ANN path when GPU isn't armed (or failed).
        if self._ensure_hnsw():
            try:
                import numpy as np
                labels, distances = self._hnsw_index.knn_query(np.asarray([query_vec], dtype=np.float32), k=min(limit, len(self._hnsw_ids)))
                out = []
                for mid, dist in zip(labels[0], distances[0]):
                    out.append({"id": int(mid), "score": max(0.0, 1.0 - float(dist)), "similarity": max(0.0, 1.0 - float(dist)), "channel": "semantic"})
                return out
            except Exception:
                pass

        if self._cpp:
            try:
                candidates = self._cpp.retrieve(query_vec, k=limit)
                if candidates:
                    out = []
                    for c in candidates:
                        cpp_id = c["id"]
                        mem_id = self._cpp_id_map.get(cpp_id, cpp_id)
                        sim = float(c.get("similarity", c.get("score", 0.0)))
                        out.append({"id": int(mem_id), "score": sim, "similarity": sim, "channel": "semantic"})
                    return out
            except Exception:
                pass

        scored = []
        for mem in self.store.get_all():
            sim = self._cosine_similarity(query_vec, mem["embedding"])
            scored.append({"id": int(mem["id"]), "score": sim, "similarity": sim, "channel": "semantic"})
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def _recent_semantic_candidates(self, query_vec: list[float], pool: int, limit: int) -> list[dict]:
        """Top-N newest memories scored by query similarity.

        Different intent from `search_temporal` (chronological grab regardless
        of query) and from `_semantic_candidates` (corpus-wide, recency-blind).
        This pulls the freshest `pool` memories then ranks them by cosine to
        the query — the slice that operators almost always mean when they ask
        about "current state", "status", or anything implying *now*.

        Without this channel, status-style queries get drowned by months of
        old memories that happen to mention the same topic with denser
        keyword saturation. The 193k-corpus has many such old hits per topic;
        a single fresh status memory from today loses the global rank race
        even though it is what the operator wants.

        Engine-internal synthesis (`derived:cluster` from the Insight phase)
        is excluded at the SQL level: each Insight cycle emits up to ~50
        cluster summaries every few minutes, which would otherwise dominate
        the freshest-N pool and bury the actual user-and-assistant turns
        (curated facts, ops memos, post-decision distillates) the operator
        meant to retrieve. derived:cluster is searchable via the semantic
        channel; recency is reserved for non-synthetic, recently-touched
        signal.
        """
        if pool <= 0 or limit <= 0:
            return []
        rows = self.store.recent_semantic_pool(int(pool),
                                               exclude_label_prefix="derived:cluster")
        scored: list[dict] = []
        for r in rows:
            vec = r.get("embedding") or []
            if not vec:
                continue
            sim = self._cosine_similarity(query_vec, vec)
            scored.append({
                "id": int(r["id"]),
                "score": float(sim),
                "similarity": float(sim),
                "channel": "recency",
            })
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def _parallel_retrieve(self, query: str, query_vec: list[float], limit: int, now: float) -> dict[str, list[dict]]:
        # Kept intentionally stdlib. SQLite reads are cheap and each lexical method
        # opens its own reader, so this can be parallelized later without API churn.
        #
        # `recency` channel: top-N newest memories ranked by query similarity.
        # Distinct from `temporal` (chronological grab, query-blind) and
        # `semantic` (corpus-wide, recency-blind). Survives `lean` mode
        # (where temporal is zeroed) — operators asking "status of X" almost
        # always mean the freshest matching memory, not the densest one.
        # Pool size 200 ≈ last few hours of activity on the 193k corpus;
        # fresh status updates win against six months of old keyword hits.
        return {
            "semantic": self._semantic_candidates(query, query_vec, limit),
            "recency": self._recent_semantic_candidates(query_vec, pool=200, limit=limit),
            "bm25": self.store.search_bm25(query, limit),
            "entity": self.store.search_entity(query, limit),
            "temporal": self.store.search_temporal(query, limit, now),
        }

    def _rrf_fuse(self, channels: dict[str, list[dict]]) -> dict[int, dict]:
        fused: dict[int, dict] = {}
        for channel, results in channels.items():
            weight = float(self._channel_weights.get(channel, 1.0))
            for rank, r in enumerate(results, start=1):
                mem_id = int(r["id"])
                contrib = weight / (self._rrf_k + rank)
                item = fused.setdefault(mem_id, {"id": mem_id, "fused_score": 0.0, "channel_scores": {}, "semantic_similarity": 0.0})
                item["fused_score"] += contrib
                item["channel_scores"][channel] = item["channel_scores"].get(channel, 0.0) + contrib
                if channel == "semantic":
                    item["semantic_similarity"] = max(float(item.get("semantic_similarity", 0.0)), float(r.get("similarity", r.get("score", 0.0)) or 0.0))
        return fused

    def _get_reranker(self):
        """Lazy cross-encoder reranker. Optional and fail-closed."""
        if self._reranker_failed:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
            model_name = os.environ.get("NEURAL_MEMORY_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            load_path = _resolve_hf_snapshot(model_name) or model_name
            self._reranker = CrossEncoder(load_path)
            return self._reranker
        except Exception:
            self._reranker_failed = True
            self._reranker = None
            return None

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-min(x, 60.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(x, -60.0))
        return z / (1.0 + z)

    def _rerank_results(self, query: str, results: list[dict], k: int) -> list[dict]:
        """Rerank top candidates with a cross-encoder, preserving fallback order on failure."""
        if len(results) <= 1:
            return results
        reranker = self._get_reranker()
        if reranker is None:
            return results
        cutoff = min(len(results), max(k * 3, k, 12))
        head = results[:cutoff]
        tail = results[cutoff:]
        pairs = [(query, (r.get("content") or "")[:2048]) for r in head]
        try:
            raw_scores = reranker.predict(pairs)
        except Exception:
            self._reranker_failed = True
            return results
        try:
            raw_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        except TypeError:
            raw_list = [float(raw_scores)]
        old_max = max((float(r.get("relevance", r.get("combined", 0.0)) or 0.0) for r in head), default=1.0)
        if old_max <= 0:
            old_max = 1.0
        scored = []
        for r, raw in zip(head, raw_list):
            item = dict(r)
            ce = self._sigmoid(float(raw))
            old = float(item.get("relevance", item.get("combined", 0.0)) or 0.0) / old_max
            combined = 0.65 * ce + 0.35 * old
            item["rerank_score"] = round(ce, 6)
            item["pre_rerank_relevance"] = item.get("relevance", item.get("combined", 0.0))
            item["relevance"] = round(combined, 6)
            item["combined"] = round(combined, 6)
            scored.append(item)
        scored.sort(key=lambda x: -float(x.get("relevance", 0.0)))
        return scored + tail

    # ── ColBERT late-interaction helpers ──────────────────────────────────
    def _dae_score_candidates(
        self,
        query_vec: "list[float]",
        fused: "dict[int, dict]",
        cap: int = 200,
    ) -> "dict[int, float]":
        """Score the top-`cap` fused candidates via DAE-embedding cosine.

        Counterpart to `_colbert_score_candidates` but for the
        Dream-Augmented Embeddings channel: each memory has an optional
        second embedding stored in `memory_dae_embeddings` weighted
        toward its graph neighbourhood.  Candidates without a DAE row
        fall through (no penalty, no boost).

        Lazy: import dae helper inline so plain semantic recall doesn't
        pay the cost.  Restricts to the top `cap` ids by current
        fused_score — DAE is a 2nd-stage rerank, not a fresh retrieval.
        """
        head_ids = [
            mid for mid, _ in sorted(
                fused.items(), key=lambda kv: -kv[1].get("fused_score", 0.0)
            )[:cap]
        ]
        if not head_ids:
            return {}
        # Dispatch on the store. Both SQLiteStore and PostgresStore now
        # expose fetch_dae_vectors(ids) — the previous form (
        # `fetch_dae_vectors(self.store.conn, ...)`) reached into a
        # SQLite-only attribute and silently produced empty results on
        # PG, which is why engine_config.py used to force-disable DAE
        # whenever MM_DB_BACKEND=postgres.
        fetch = getattr(self.store, "fetch_dae_vectors", None)
        if fetch is None:
            return {}
        try:
            vecs = fetch(head_ids)
        except Exception as exc_dae:
            logger.warning("DAE fetch failed via %s: %s", type(self.store).__name__, exc_dae)
            return {}
        if not vecs:
            return {}
        out: "dict[int, float]" = {}
        for mid, dvec in vecs.items():
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > 0:
                out[int(mid)] = float(sim)
        return out

    def _colbert_score_candidates(
        self,
        query: str,
        fused: "dict[int, dict]",
        cap: int = 100,
    ) -> "dict[int, float]":
        """Score the top-`cap` fused candidates via BGE-M3 token-level
        max-sim. Returns {memory_id: float} for candidates that had a
        cached blob; absent ids fell through (no penalty, just no boost).

        Lazy in two ways:
          1. Imports colbert_helper only when called (no model load on
             plain semantic recall).
          2. Restricts to the top `cap` ids by current fused_score —
             ColBERT is a 2nd-stage rerank, not a fresh retrieval.
        """
        try:
            from colbert_helper import (
                encode_tokens,
                unpack_tokens,
                score_late_interaction,
            )
        except Exception:
            return {}
        head_ids = [
            mid for mid, _ in sorted(
                fused.items(), key=lambda kv: -kv[1].get("fused_score", 0.0)
            )[:cap]
        ]
        if not head_ids:
            return {}
        try:
            blobs = self.store.get_colbert_tokens_many(head_ids)
        except Exception:
            return {}
        if not blobs:
            return {}
        try:
            q_arr = encode_tokens(query)
        except Exception:
            return {}
        if q_arr is None:
            return {}
        import numpy as np
        docs: "list[Optional[np.ndarray]]" = []
        order: "list[int]" = []
        for mid in head_ids:
            blob = blobs.get(mid)
            if not blob:
                continue
            arr = unpack_tokens(blob)
            if arr is None:
                continue
            docs.append(arr.astype(np.float32))
            order.append(mid)
        if not docs:
            return {}
        scores = score_late_interaction(q_arr.astype(np.float32), docs)
        return {mid: float(s) for mid, s in zip(order, scores)}

    def recall_batch(self, queries: "list[str]", k: int = 5) -> "list[list[dict]]":
        """Batched semantic recall.

        Caller-friendly wrapper that drives `GpuRecallEngine.recall_batch`
        when CUDA is armed (one matmul, one embed-server round-trip for the
        whole batch) and falls through to per-query `recall()` otherwise.
        Used by the dream loop's REM phase to collapse 800 sequential
        recalls into a single GPU batch.

        Returns a list of result lists, parallel to `queries`. Each result
        list mirrors the shape of `recall()` enough for REM
        (`{id, similarity, ...}`).
        """
        if not queries:
            return []
        if self._gpu is not None and hasattr(self._gpu, "recall_batch"):
            try:
                out = self._gpu.recall_batch(queries, k=k)
                if out and len(out) == len(queries):
                    return out
            except Exception:
                pass
        return [self.recall(q, k=k) for q in queries]

    def recall_multi(
        self,
        queries: "list[str]",
        k: int = 10,
        fuse: bool = True,
        k_per_query: Optional[int] = None,
    ) -> "list[dict] | list[list[dict]]":
        """Multi-perspective recall: run N queries in one call and fuse results.

        When ``fuse=True`` (default) results from all queries are merged via
        Reciprocal Rank Fusion (RRF) and deduplicated, returning a flat list of
        at most *k* dicts.  Each returned dict includes a unified ``score``
        field set to the memory's RRF score.

        When ``fuse=False`` the raw per-query results are returned as
        ``list[list[dict]]`` (same shape as ``recall_batch``), with each dict
        carrying a ``score`` field copied from ``rerank_score or relevance or
        combined or similarity``.

        Args:
            queries:      List of query strings to run in parallel.
            k:            Maximum results to return (fused mode) or per query
                          (passthrough mode).
            fuse:         If True, apply RRF and return a flat deduplicated list.
            k_per_query:  Candidates fetched per query before fusion.
                          Defaults to ``2 * k`` so the fused pool is wide
                          enough to surface high-quality results for every angle.

        Returns:
            Flat ``list[dict]`` when ``fuse=True``, else ``list[list[dict]]``.
        """
        if not queries:
            return [] if fuse else []

        per_q = k_per_query if k_per_query is not None else k * 2
        # Use FULL recall() pipeline per query (ColBERT + DAE + hybrid + rerank).
        # recall_batch() is the REM fast-path — pure-semantic cosine matmul,
        # not suitable for quality multi-perspective fusion.
        per_query_results: list[list[dict]] = [
            self.recall(q, k=per_q, hybrid=True,
                        enable_colbert=True, colbert_weight=1.5,
                        enable_dae=True, dae_weight=1.0)
            for q in queries
        ]

        if not fuse:
            # Passthrough — ensure score field present.
            for qr in per_query_results:
                for r in qr:
                    r.setdefault('score', (
                        r.get('rerank_score') or
                        r.get('relevance') or
                        r.get('combined') or
                        r.get('similarity') or
                        0.0
                    ))
            return per_query_results

        # --- RRF fusion ---
        # rrf_scores[id] accumulates 1/(60 + rank) summed over all queries.
        rrf_scores: dict[int, float] = {}
        id_to_result: dict[int, dict] = {}

        for qr in per_query_results:
            for rank, r in enumerate(qr):
                mem_id = r.get('id')
                if mem_id is None:
                    continue
                rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + 1.0 / (60 + rank)
                if mem_id not in id_to_result:
                    id_to_result[mem_id] = r

        # Sort by RRF score descending, attach unified score, return top-k.
        ranked_ids = sorted(rrf_scores, key=lambda i: -rrf_scores[i])
        fused: list[dict] = []
        for mem_id in ranked_ids[:k]:
            result = dict(id_to_result[mem_id])
            result.pop('embedding', None)
            result['score'] = round(rrf_scores[mem_id], 8)
            fused.append(result)
        return fused

    def recall_with_neighbors(
        self,
        query: str,
        k: int = 10,
        depth: int = 1,
        neighbor_weight: float = 0.3,
        **recall_kwargs,
    ) -> list[dict]:
        """Graph-augmented recall: seed set from ``recall()`` + connected memories.

        After fetching a seed set of top-``k*2`` matches, traverse each seed's
        graph connections (1 or 2 hops) and merge in linked memories that were
        not in the seed set.  Neighbor scores decay multiplicatively with hop
        distance and connection weight:

            neighbor_score = neighbor_weight * source_score * connection_weight

        For ``depth=2`` the decay is applied again for each second-hop edge:

            hop2_score = neighbor_weight * hop1_score * connection_weight

        All results are deduplicated by id and sorted descending by score.
        Each dict carries a ``via`` field — ``"direct"`` for seed memories and
        ``"neighbor:hop1"`` / ``"neighbor:hop2"`` for graph-expanded ones — so
        consumers can distinguish how a memory was surfaced.

        Args:
            query:            Query string for the initial seed recall.
            k:                Maximum results to return.
            depth:            Hop depth for graph traversal (1 or 2).
            neighbor_weight:  Base weight multiplier for neighbour scores.
            **recall_kwargs:  Extra kwargs forwarded to ``recall()``
                              (e.g. ``mmr_lambda``, ``score_floor``).

        Returns:
            Flat ``list[dict]`` of at most *k* results, sorted by score desc.
        """
        # --- Seed set ---
        seeds = self.recall(query, k=k * 2, **recall_kwargs)
        seen_ids: dict[int, dict] = {}  # id → result dict

        for r in seeds:
            mem_id = r.get('id')
            if mem_id is None:
                continue
            r = dict(r)
            r.pop('embedding', None)
            r.setdefault('score', (
                r.get('rerank_score') or
                r.get('relevance') or
                r.get('combined') or
                r.get('similarity') or
                0.0
            ))
            r['via'] = 'direct'
            seen_ids[mem_id] = r

        # --- Graph traversal ---
        def _expand(frontier: "dict[int, float]", hop_label: str) -> "dict[int, dict]":
            """Expand one hop from frontier {id: score} and return new nodes."""
            new_nodes: dict[int, dict] = {}
            for source_id, source_score in frontier.items():
                try:
                    conns = self.connections(source_id)
                except Exception:
                    continue
                for c in conns:
                    neighbour_id = c.get('id')
                    if neighbour_id is None or neighbour_id in seen_ids:
                        continue
                    conn_weight = float(c.get('weight', 1.0))
                    n_score = neighbor_weight * source_score * conn_weight
                    if neighbour_id in new_nodes:
                        if n_score <= new_nodes[neighbour_id]['score']:
                            continue
                    try:
                        mem = self.store.get(neighbour_id, include_embedding=False)
                    except Exception:
                        mem = None
                    if mem is None:
                        continue
                    new_nodes[neighbour_id] = {
                        'id': neighbour_id,
                        'label': mem.get('label', ''),
                        'content': mem.get('content', ''),
                        'score': round(n_score, 8),
                        'via': hop_label,
                        'connections': [],
                        'created_at': mem.get('created_at'),
                        'last_accessed': mem.get('last_accessed'),
                        'access_count': mem.get('access_count', 0),
                    }
            return new_nodes

        # Hop 1
        frontier_scores = {mid: r['score'] for mid, r in seen_ids.items()}
        hop1_nodes = _expand(frontier_scores, 'neighbor:hop1')
        seen_ids.update(hop1_nodes)

        # Hop 2
        if depth >= 2:
            hop2_frontier = {mid: r['score'] for mid, r in hop1_nodes.items()}
            hop2_nodes = _expand(hop2_frontier, 'neighbor:hop2')
            seen_ids.update(hop2_nodes)

        # Sort descending by score, cap at k.
        ranked = sorted(seen_ids.values(), key=lambda r: -r.get('score', 0.0))
        return ranked[:k]

    def recall(
        self,
        query: str,
        k: int = 5,
        temporal_weight: float = 0.2,
        query_vec: Optional[list[float]] = None,
        touch: bool = True,
        hybrid: Optional[bool] = None,
        rerank: Optional[bool] = None,
        at_time: Optional[float] = None,
        mmr_lambda: Optional[float] = None,
        score_floor: Optional[float] = None,
        supersedes_traversal: bool = True,
        # score_percentile: drop the bottom X fraction of candidates BY
        # RANK before truncating to k. Calibrated [0,1] alternative to
        # score_floor (which operates on the RRF-derived raw relevance
        # scale ~[0, 0.05] and is therefore badly calibrated — codex
        # 2026-04-28 benchmark caught that score_floor>=0.2 nukes
        # results). e.g. score_percentile=0.5 keeps the top half of
        # ranked candidates. Off by default (0.0).
        score_percentile: Optional[float] = None,
        # enable_colbert: explicitly turn the ColBERT-style late-interaction
        # rerank channel on or off for this call. None (default) inherits the
        # constructor's `colbert` channel weight (0 = off). True forces the
        # channel even for memories without cached blobs (they simply get
        # 0 contribution and don't poison the fusion).
        enable_colbert: Optional[bool] = None,
        # colbert_weight: per-call override for the colbert channel
        # contribution. None inherits the constructor channel weight.
        # Set to 0.0 to disable for this call without touching the
        # constructor; >1.0 emphasises the rerank.
        colbert_weight: Optional[float] = None,
        # enable_dae / dae_weight: same shape as ColBERT but for the
        # Dream-Augmented Embeddings channel.  None (default) inherits
        # the constructor `dae` channel weight (0 = off, scaffolding
        # only).  True forces the channel even when the constructor has
        # it off.  See python/dae.py for the math.
        enable_dae: Optional[bool] = None,
        dae_weight: Optional[float] = None,
        # include_echoes: when False (default), strip per-turn conversation
        # logs (`auto:turn:*`, `auto:claude:*`, `auto:hermes:*`, anything
        # starting with `auto:`) from the candidate set BEFORE relevance
        # scoring. The 193k-memory corpus is ~98% auto-saved conversation
        # echoes; these crowd out facts because their semantic embedding
        # is dominated by the verbatim user question, not the synthesised
        # answer. Recall callers (Hermes, Claude, MCP) want facts and
        # synthesised insights only — set True only for session-replay
        # use cases that genuinely need raw turn content.
        include_echoes: bool = False,
    ) -> list[dict]:
        if k <= 0:
            return []
        query_vec = query_vec or self.embedder.embed(query)
        now = time.time()
        limit = max(k * 4, self._retrieval_candidates)
        # Echo filter is cheap up-front: pull more candidates from each
        # channel so that after we drop user-question echoes we still have
        # enough facts to fill k. Without this, queries like "trailer
        # status" (where the corpus has tons of `auto:hermes:*:u` echoes
        # of the exact same question) would degenerate to an empty result
        # set.
        if not include_echoes:
            limit = limit * 2
        if hybrid is None:
            # `lean`/`trim` are cost-conscious skynet variants that zero
            # dead-weight channels per benchmark findings — still hybrid.
            hybrid = self._retrieval_mode in {"hybrid", "advanced", "skynet", "lean", "trim"}

        if hybrid:
            channels = self._parallel_retrieve(query, query_vec, limit, now)
        else:
            channels = {"semantic": self._semantic_candidates(query, query_vec, limit)}

        fused = self._rrf_fuse(channels)
        if not fused:
            return []

        ppr_scores: dict[int, float] = {}
        if self._think_engine == "ppr" or hybrid:
            seed_scores = {mid: data["fused_score"] for mid, data in sorted(fused.items(), key=lambda kv: -kv[1]["fused_score"])[: min(12, len(fused))]}
            ppr_scores = self._ppr_scores(seed_scores, alpha=self._ppr_alpha, iters=self._ppr_iters, hops=self._ppr_hops)
            for mid, score in sorted(ppr_scores.items(), key=lambda kv: -kv[1])[:limit]:
                fused.setdefault(mid, {"id": mid, "fused_score": 0.0, "channel_scores": {}, "semantic_similarity": 0.0})
                fused[mid]["channel_scores"]["ppr"] = score * self._channel_weights.get("ppr", 0.55) / self._rrf_k

        # ── ColBERT late-interaction rerank ──────────────────────────────
        # BGE-M3 already emits token-level embeddings; we pre-cache the
        # top-32 per memory in the `colbert_tokens` BLOB column. At
        # recall, score the top-100 fused candidates with ColBERT
        # max-sim and add as a new fusion channel. Default-off; enabled
        # per call via enable_colbert=True or globally via the `colbert`
        # channel weight in skynet/advanced
        # presets. Memories without a cached blob simply skip this
        # channel; relevance still comes through the other six.
        cb_weight = (
            float(colbert_weight) if colbert_weight is not None
            else float(self._channel_weights.get("colbert", 0.0) or 0.0)
        )
        cb_on = (
            (enable_colbert is True)
            or (enable_colbert is None and cb_weight > 0.0)
        )
        # Pro feature gate.  Community / Lite installs simply skip the
        # ColBERT fusion channel; the other six channels still produce
        # the published R@5 = 0.96 hybrid number.
        if cb_on and not has_feature("colbert"):
            cb_on = False
        if cb_on and cb_weight > 0.0 and fused:
            colbert_scores = self._colbert_score_candidates(query, fused)
            if colbert_scores:
                # Rank-based RRF contribution mirrors the other channels in
                # _rrf_fuse: score-by-rank into the same RRF denominator,
                # weighted by the channel weight. This keeps ColBERT
                # additive — it can lift a precise candidate from rank 50
                # to top-5 without overwhelming the multi-channel signal.
                ordered = sorted(colbert_scores.items(), key=lambda kv: -kv[1])
                for rank, (mid, raw) in enumerate(ordered, start=1):
                    if mid not in fused:
                        continue
                    contrib = cb_weight / (self._rrf_k + rank)
                    item = fused[mid]
                    item["fused_score"] = item.get("fused_score", 0.0) + contrib
                    item["channel_scores"]["colbert"] = (
                        item["channel_scores"].get("colbert", 0.0) + contrib
                    )
                    item["colbert_raw"] = float(raw)

        # DAE channel — same shape as ColBERT, scored against the
        # `memory_dae_embeddings` table populated by `dae_bulk_compute`
        # (bench path) or, eventually, the NREM phase (production path
        # — pending bench validation).  Pro-gated.  Per-call kwargs
        # override the constructor channel weight.
        dae_weight_v = (
            float(dae_weight) if dae_weight is not None
            else float(self._channel_weights.get("dae", 0.0) or 0.0)
        )
        dae_on = (
            (enable_dae is True)
            or (enable_dae is None and dae_weight_v > 0.0)
        )
        if dae_on and not has_feature("dae"):
            dae_on = False
        if dae_on and dae_weight_v > 0.0 and fused:
            dae_scores = self._dae_score_candidates(query_vec, fused)
            if dae_scores:
                ordered = sorted(dae_scores.items(), key=lambda kv: -kv[1])
                for rank, (mid, raw) in enumerate(ordered, start=1):
                    if mid not in fused:
                        continue
                    contrib = dae_weight_v / (self._rrf_k + rank)
                    item = fused[mid]
                    item["fused_score"] = item.get("fused_score", 0.0) + contrib
                    item["channel_scores"]["dae"] = (
                        item["channel_scores"].get("dae", 0.0) + contrib
                    )
                    item["dae_raw"] = float(raw)

        mems = self.store.get_many(list(fused.keys()), include_embedding=True)

        # ── Echo filter ──────────────────────────────────────────────────
        # Drop pure user-question echoes — labels ending in `:u` are
        # `auto:hermes:<session>:t<N>:u` memories holding the verbatim
        # user input only. Same goes for content-empty user turn dumps.
        # Everything else (auto:turn:* full Claude-Code dumps with both
        # USER and ASSISTANT sections, auto:hermes:*:a assistant turns,
        # auto:claude:* legacy per-turn, derived:cluster summaries, all
        # curated labels) carries real signal and stays.
        # We do this BEFORE the per-memory connection fetch so SQL
        # round-trips are spent only on results that will actually
        # be returned.
        if not include_echoes:
            # User-question echoes carry the role marker `:u` — either as the
            # last segment (`auto:hermes:<sid>:t<N>:u`) or before a paragraph
            # suffix when the body got chunked (`...:t<N>:u:p<M>`). Match both
            # by tokenising on `:` and checking for a bare `u` segment that
            # is either the last token or directly precedes a `p<digits>`
            # paragraph token. This is tighter than substring matching, which
            # would false-positive on labels like `user:something`.
            import re
            _PARA_RE = re.compile(r"^p\d+$")
            def _is_user_echo(label: str) -> bool:
                parts = label.split(":")
                for i, seg in enumerate(parts):
                    if seg != "u":
                        continue
                    if i == len(parts) - 1:
                        return True
                    if i == len(parts) - 2 and _PARA_RE.match(parts[i + 1]):
                        return True
                return False

            fact_ids: list[int] = []
            for mid, mem in mems.items():
                label = (mem.get("label") or "")
                if _is_user_echo(label):
                    continue
                fact_ids.append(mid)
            mems = {mid: mems[mid] for mid in fact_ids}
            fused = {mid: fused[mid] for mid in fact_ids if mid in fused}
            if not fused:
                return []

        # Pre-fetch connections per result-memory and collect every neighbour
        # id we'll need to render. The previous code did one get_connections
        # call PLUS up to 3 single-row get() calls per result memory — a
        # 4N SQL round-trip pattern that dominated recall latency at scale.
        # One get_many on the whole neighbour set replaces 3*N queries.
        per_mem_conns: dict[int, list] = {}
        all_neighbour_ids: set[int] = set()
        for mem_id in fused.keys():
            conns = self.store.get_connections(mem_id, at_time=at_time)[:3]
            per_mem_conns[mem_id] = conns
            for c in conns:
                other = c["target"] if c["source"] == mem_id else c["source"]
                all_neighbour_ids.add(int(other))
        neighbour_mems = (
            self.store.get_many(list(all_neighbour_ids), include_embedding=False)
            if all_neighbour_ids else {}
        )

        results = []
        for mem_id, data in fused.items():
            mem = mems.get(mem_id)
            if not mem:
                continue
            sim = float(data.get("semantic_similarity") or 0.0)
            if sim == 0.0 and mem.get("embedding"):
                sim = self._cosine_similarity(query_vec, mem["embedding"])
            temporal_score = self._compute_temporal_score_from_mem(mem, now)
            salience_factor = self._effective_salience(mem, now)
            ppr = float(ppr_scores.get(mem_id, 0.0))
            relevance = (
                float(data.get("fused_score", 0.0))
                + temporal_weight * 0.10 * temporal_score
                + self._channel_weights.get("salience", 0.25) * 0.05 * salience_factor
                + self._channel_weights.get("ppr", 0.55) * 0.10 * ppr
            )
            connected = []
            for c in per_mem_conns.get(mem_id, ()):
                other = c["target"] if c["source"] == mem_id else c["source"]
                other_mem = neighbour_mems.get(int(other))
                if other_mem:
                    connected.append({
                        "id": other,
                        "label": other_mem["label"],
                        "weight": round(float(c["weight"]), 4),
                        "type": c.get("type", "similar"),
                    })
            results.append({
                "id": mem_id,
                "label": mem.get("label", ""),
                "content": mem.get("content", ""),
                "similarity": round(sim, 4),
                "temporal_score": round(temporal_score, 4),
                "salience_factor": round(salience_factor, 4),
                "ppr_score": round(ppr, 4),
                "colbert_score": round(float(data.get("colbert_raw", 0.0) or 0.0), 4),
                "combined": round(relevance, 6),
                "relevance": round(relevance, 6),
                "channel_scores": {ch: round(float(v), 6) for ch, v in data.get("channel_scores", {}).items()},
                "connections": connected,
                "created_at": mem.get("created_at"),
                "last_accessed": mem.get("last_accessed"),
                "access_count": mem.get("access_count", 0),
            })

        results.sort(key=lambda x: -x["relevance"])
        # Rerank default: explicit kwarg wins; otherwise inherit constructor
        # value, but for the latency-tolerant modes (`advanced` / `skynet`)
        # auto-promote to True. Lean / trim / hybrid / semantic stay opt-in
        # because their whole point is short p50. Cherry-picked policy
        # change from upstream PR #5 brainstorm — the cross-encoder lift on
        # recall@5 (~+10pp on noisy queries) is the whole reason the
        # modes exist.
        if rerank is None:
            if self._retrieval_mode in ("advanced", "skynet"):
                use_rerank = True
            else:
                use_rerank = self._rerank
        else:
            use_rerank = bool(rerank)
        if use_rerank:
            results = self._rerank_results(query, results, k)

        # Hard noise floor — drop sub-floor results before MMR/truncation so
        # garbage queries return [] instead of weak top-k that operators
        # mistake for signal.
        floor = self._recall_score_floor if score_floor is None else float(score_floor or 0.0)
        if floor > 0.0:
            results = [r for r in results if r.get("relevance", 0.0) >= floor]

        # Calibrated percentile alternative: keep only the top (1-pct)
        # fraction by rank. Operates on the sorted result list, not on
        # raw RRF scores, so it's well-defined regardless of the
        # underlying relevance scale. e.g. score_percentile=0.5 → keep
        # top half. Combines with score_floor — both are applied if set.
        # Falls back to the constructor default when the per-call kwarg
        # is None (matches mmr_lambda / score_floor behaviour).
        pct = (
            self._recall_score_percentile if score_percentile is None
            else float(score_percentile or 0.0)
        )
        if pct > 0.0:
            keep_n = max(1, int(round(len(results) * (1.0 - min(1.0, pct)))))
            results = results[:keep_n]

        # MMR diversification: trade marginal relevance for non-redundancy in
        # the returned set. Greedy O(k * |candidates|), bounded by k * limit.
        # Skipped when mmr_lambda is at its default 0.0 (pure relevance) or
        # the candidate pool already has <2 items.
        eff_lambda = self._mmr_lambda if mmr_lambda is None else max(0.0, min(1.0, float(mmr_lambda or 0.0)))
        if eff_lambda > 0.0 and len(results) > 1 and k > 1:
            results = self._mmr_rerank(results, mems, eff_lambda, k)

        # ── SUPERSEDES traversal ─────────────────────────────────────────
        # For each result X that has an outgoing 'supersedes' edge to Y
        # (meaning X is the older/superseded version):
        #   1. Demote X's score by 0.5 (it's stale — prefer the newer one).
        #   2. Elevate Y's score to at least X's pre-demotion score.
        #   3. If Y is not already in results, fetch it and add it.
        #   4. Tag X with 'superseded_by': Y's id.
        if supersedes_traversal and results:
            result_ids = {r["id"] for r in results}
            # For each result, check for outgoing supersedes edges.
            # A memory is "the older one" when it is the SOURCE of a
            # supersedes edge (source=older, target=newer — established by
            # _phase_supersedes and _detect_conflicts).
            superseded_map: dict[int, int] = {}  # old_id → new_id
            for r in results:
                mid = r["id"]
                try:
                    conns = self.store.get_connections(mid)
                    for c in conns:
                        if (c.get("edge_type") or c.get("type", "")) != "supersedes":
                            continue
                        # Directed: source=older, target=newer
                        if c["source"] == mid:
                            newer_id = int(c["target"])
                            superseded_map[mid] = newer_id
                except Exception:
                    pass

            if superseded_map:
                # Collect newer memory ids not yet in results.
                missing_newer = {nid for nid in superseded_map.values() if nid not in result_ids}
                # Fetch missing newer memories in one SQL call.
                extra_mems: dict[int, dict] = {}
                if missing_newer:
                    try:
                        extra_mems = self.store.get_many(list(missing_newer), include_embedding=True)
                    except Exception:
                        pass

                # Build id → result dict for score edits.
                id_to_result: dict[int, dict] = {r["id"]: r for r in results}

                # Materialise newer memories that aren't already results.
                for nid, nmem in extra_mems.items():
                    if not nmem:
                        continue
                    nsim = 0.0
                    if nmem.get("embedding") and query_vec:
                        nsim = self._cosine_similarity(query_vec, nmem["embedding"])
                    new_entry: dict = {
                        "id": nid,
                        "label": nmem.get("label", ""),
                        "content": nmem.get("content", ""),
                        "similarity": round(nsim, 4),
                        "temporal_score": 0.0,
                        "salience_factor": 1.0,
                        "ppr_score": 0.0,
                        "colbert_score": 0.0,
                        "combined": 0.0,
                        "relevance": 0.0,
                        "channel_scores": {},
                        "connections": [],
                        "created_at": nmem.get("created_at"),
                        "last_accessed": nmem.get("last_accessed"),
                        "access_count": nmem.get("access_count", 0),
                        "superseded_by": None,
                    }
                    id_to_result[nid] = new_entry
                    results.append(new_entry)

                for old_id, new_id in superseded_map.items():
                    old_r = id_to_result.get(old_id)
                    new_r = id_to_result.get(new_id)
                    if old_r is None:
                        continue
                    pre_demotion = float(old_r.get("relevance") or old_r.get("combined") or 0.0)
                    demoted = max(0.0, pre_demotion - 0.5)
                    old_r["relevance"] = round(demoted, 6)
                    old_r["combined"] = round(demoted, 6)
                    old_r["superseded_by"] = new_id
                    if new_r is not None:
                        cur_new = float(new_r.get("relevance") or new_r.get("combined") or 0.0)
                        elevated = max(cur_new, pre_demotion + 0.01)
                        new_r["relevance"] = round(elevated, 6)
                        new_r["combined"] = round(elevated, 6)

                results.sort(key=lambda x: -x.get("relevance", 0.0))

        final = results[:k]
        if touch:
            for r in final:
                try:
                    self.store.touch(r["id"])
                except Exception:
                    pass
        # Unified convenience field — highest-precedence non-zero score so
        # consumers don't have to check the rerank/relevance/combined/similarity
        # cascade themselves.
        for r in final:
            r.setdefault('score', (
                r.get('rerank_score') or
                r.get('relevance') or
                r.get('combined') or
                r.get('similarity') or
                0.0
            ))
        return final

    def _mmr_rerank(
        self,
        ranked: list[dict],
        mems: dict[int, dict],
        lam: float,
        k: int,
    ) -> list[dict]:
        """Greedy Maximal Marginal Relevance reordering.

        MMR(d) = lam * relevance(d) - (1-lam) * max_{d' in selected} sim(d, d')

        - relevance: the precomputed combined score (already in [0,1]-ish range)
        - sim: cosine between memory embeddings (0 if either embedding is
          missing or dim-mismatched, see _cosine_similarity)

        We walk the ranked list, picking the candidate that maximises MMR
        against the already-selected set. Greedy is the standard MMR
        formulation (Carbonell & Goldstein 1998); exact MMR is NP-hard.
        """
        if lam >= 1.0:
            return ranked
        # Normalise relevance to [0,1] so the diversity term is comparable.
        max_rel = max((r.get("relevance", 0.0) for r in ranked), default=1.0) or 1.0
        pool = list(ranked)
        selected: list[dict] = []
        # Cache embeddings keyed by memory id to avoid repeated dict lookups.
        embed_of = {mid: m.get("embedding") or [] for mid, m in mems.items()}
        # Always seed with the top-relevance pick — MMR's first iteration is
        # equivalent to argmax(relevance) since selected is empty.
        first = pool.pop(0)
        selected.append(first)
        # Running max-similarity-to-selected for every remaining candidate.
        # Updated incrementally as we add to `selected` — saves the O(|sel|)
        # inner max() call per candidate per outer iteration. Total work
        # drops from O(k * |pool| * |selected|) to O(k * |pool|).
        max_sim_to_sel: dict[int, float] = {}
        first_emb = embed_of.get(first["id"], [])
        for cand in pool:
            cand_emb = embed_of.get(cand["id"], [])
            if cand_emb and first_emb:
                max_sim_to_sel[cand["id"]] = self._cosine_similarity(cand_emb, first_emb)
            else:
                max_sim_to_sel[cand["id"]] = 0.0
        while pool and len(selected) < k:
            best_idx = -1
            best_score = -float("inf")
            for i, cand in enumerate(pool):
                rel = cand.get("relevance", 0.0) / max_rel
                div = max_sim_to_sel.get(cand["id"], 0.0)
                score = lam * rel - (1.0 - lam) * div
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx < 0:
                break
            picked = pool.pop(best_idx)
            selected.append(picked)
            # Update each remaining candidate's running max against the
            # newly-picked item only — O(|pool|) per outer iter.
            picked_emb = embed_of.get(picked["id"], [])
            if picked_emb:
                for cand in pool:
                    cand_emb = embed_of.get(cand["id"], [])
                    if not cand_emb:
                        continue
                    sim = self._cosine_similarity(cand_emb, picked_emb)
                    if sim > max_sim_to_sel.get(cand["id"], 0.0):
                        max_sim_to_sel[cand["id"]] = sim
        # Append leftovers (preserving original ranking) so consumers asking
        # for k > pool_size or downstream code that walks past k still gets
        # the full ordering rather than a truncated tail.
        return selected + pool

    # -- thinking / graph -----------------------------------------------------

    # GPU-accelerated PPR — when CUDA is available (self._gpu is loaded), the
    # whole-graph adjacency matrix gets pushed to GPU memory once and PPR
    # becomes a sequence of sparse mat-vec multiplies. On a 1M-edge graph this
    # is ~3 orders of magnitude faster than the pure-Python dict-walk in
    # _ppr_scores. Per "GPU > CPU IMMER" policy: when CUDA is selected and
    # the build fails, raise loudly instead of silently falling back to CPU
    # — silent fallbacks on a PRO-tier customer are a worse failure than a
    # crash that surfaces the bug.
    def _build_gpu_ppr_adjacency(self) -> bool:
        """Build a row-stochastic sparse adjacency tensor on the GPU device.
        Returns True on success, False if torch/CUDA unavailable. Caller
        decides whether to fall back or raise.

        Device selection is independent of GpuRecallEngine: PPR only
        needs a CUDA handle + the graph adjacency, not the embedding
        tensor. The previous form required `self._gpu` to be armed,
        which is only true on SQLite (GpuRecallEngine loads from the
        SQLite-side gpu_cache). Under MM_DB_BACKEND=postgres the
        embedding tensor stays unloaded, so this gate was the reason
        DAE compute fell through to the per-edge SQL path and CPU.
        """
        try:
            import torch
        except ImportError:
            return False
        device = getattr(self._gpu, "_device", None) if self._gpu else None
        if device is None or getattr(device, "type", None) != "cuda":
            # No armed GpuRecallEngine — pick the CUDA device directly
            # if torch can see one. Same effective hardware, just
            # uncoupled from the recall-side cache.
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
            except Exception:
                device = None
        if device is None or getattr(device, "type", None) != "cuda":
            return False

        rows = self.store.weighted_edges()
        if not rows:
            self._gpu_ppr_adj = None
            self._gpu_ppr_node_to_idx = {}
            self._gpu_ppr_idx_to_node = []
            self._gpu_ppr_dirty = False
            return True

        # Row-normalize undirected: each edge (s,t,w) contributes
        #   A[s_idx, t_idx] += w / out_deg(s)
        #   A[t_idx, s_idx] += w / out_deg(t)
        node_to_idx: dict[int, int] = {}
        for s, t, _w in rows:
            if s not in node_to_idx:
                node_to_idx[s] = len(node_to_idx)
            if t not in node_to_idx:
                node_to_idx[t] = len(node_to_idx)
        n = len(node_to_idx)

        out_weight = [0.0] * n
        for s, t, w in rows:
            out_weight[node_to_idx[s]] += float(w)
            out_weight[node_to_idx[t]] += float(w)

        src_idx: list[int] = []
        dst_idx: list[int] = []
        vals: list[float] = []
        for s, t, w in rows:
            si = node_to_idx[s]
            ti = node_to_idx[t]
            ow_s = out_weight[si] or 1e-12
            ow_t = out_weight[ti] or 1e-12
            # column = source (we'll do A @ p where A is column-stochastic →
            # use indices [target, source] so the mat-vec spreads source's
            # mass into its targets).
            src_idx.append(ti); dst_idx.append(si); vals.append(float(w) / ow_s)
            src_idx.append(si); dst_idx.append(ti); vals.append(float(w) / ow_t)

        idx_tensor = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=device)
        val_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
        self._gpu_ppr_adj = torch.sparse_coo_tensor(
            idx_tensor, val_tensor, (n, n)
        ).coalesce()
        self._gpu_ppr_node_to_idx = node_to_idx
        self._gpu_ppr_idx_to_node = [0] * n
        for nid, i in node_to_idx.items():
            self._gpu_ppr_idx_to_node[i] = nid
        self._gpu_ppr_dirty = False
        return True

    def _invalidate_gpu_ppr_adjacency(self) -> None:
        """Mark the cached GPU adjacency dirty. Called by writes that change
        edge weights (batch_strengthen, batch_weaken, prune_weak, add_connection)."""
        self._gpu_ppr_dirty = True

    def _ppr_scores_gpu(
        self,
        seeds: dict[int, float],
        alpha: float = 0.15,
        iters: int = 20,
    ) -> dict[int, float]:
        """Personalized PageRank on the GPU. Falls back to CPU only if
        torch/CUDA isn't available; in CUDA-strict mode the caller decides
        what to do with a None return."""
        if not seeds:
            return {}
        try:
            import torch
        except ImportError:
            return None  # caller falls back

        # Lazy build / rebuild on dirty
        if (
            getattr(self, "_gpu_ppr_adj", None) is None
            or getattr(self, "_gpu_ppr_dirty", True)
        ):
            if not self._build_gpu_ppr_adjacency():
                return None

        if self._gpu_ppr_adj is None:
            return {}

        node_to_idx = self._gpu_ppr_node_to_idx
        idx_to_node = self._gpu_ppr_idx_to_node
        n = len(idx_to_node)
        device = self._gpu_ppr_adj.device

        # Build personalization vector. Seeds whose nodes have no edges in the
        # graph are ignored (they have no outgoing rank to spread anyway).
        total = sum(max(0.0, float(v)) for v in seeds.values()) or 1.0
        personalization = torch.zeros(n, device=device, dtype=torch.float32)
        any_seed_present = False
        for nid, w in seeds.items():
            idx = node_to_idx.get(int(nid))
            if idx is not None:
                personalization[idx] = max(0.0, float(w)) / total
                any_seed_present = True
        if not any_seed_present:
            return {}

        p = personalization.clone()
        for _ in range(max(1, iters)):
            # column-stochastic adj × p → spreads each node's mass to its
            # neighbours according to its outgoing-weight share.
            spread = torch.sparse.mm(
                self._gpu_ppr_adj, p.unsqueeze(1)
            ).squeeze(1)
            p = alpha * personalization + (1.0 - alpha) * spread
            s = float(p.sum())
            if s > 0:
                p = p / s

        p_cpu = p.cpu().numpy()
        max_score = float(p_cpu.max()) if p_cpu.size else 0.0
        if max_score <= 0:
            return {}

        scores: dict[int, float] = {}
        # Threshold at 1e-9 so the dict isn't 200k entries of float dust.
        cutoff = max_score * 1e-6
        seed_ids = {int(s) for s in seeds.keys()}
        for i in range(n):
            v = float(p_cpu[i])
            if v <= cutoff:
                continue
            nid = idx_to_node[i]
            if nid in seed_ids and v <= 0:
                continue
            scores[nid] = v / max_score
        return scores

    def _ppr_scores(self, seeds: dict[int, float], alpha: float = 0.15, iters: int = 20, hops: int = 2) -> dict[int, float]:
        if not seeds:
            return {}
        # GPU-first dispatch: when CUDA recall is armed (self._gpu loaded),
        # PPR runs on the GPU. The CPU dict-walk below is the free-tier
        # path or a torch-import-failure fallback. Per "GPU > CPU IMMER":
        # if the GPU build fails on a CUDA-strict deployment, we'd rather
        # crash loudly here than silently grind the CPU for hours.
        if self._gpu is not None:
            gpu_scores = self._ppr_scores_gpu(seeds, alpha=alpha, iters=iters)
            if gpu_scores is not None:
                return gpu_scores
            # gpu_scores is None when torch is missing or the device
            # isn't CUDA. We deliberately fall through here and let the
            # CPU path serve — the GpuRecallEngine load above already
            # logged the device, so any operator looking at the dream
            # daemon log can correlate.
        nodes = set(int(s) for s in seeds)
        frontier = set(nodes)
        for _ in range(max(0, hops)):
            nxt = set()
            for node in list(frontier):
                if self._ensure_node(node):
                    nxt.update(self._graph_nodes.get(node, {}).get("connections", {}).keys())
            nxt -= nodes
            nodes |= nxt
            frontier = nxt
            if not frontier:
                break
        if not nodes:
            return {}
        adj: dict[int, dict[int, float]] = {}
        for node in nodes:
            self._ensure_node(node)
            conns = {int(n): float(w) for n, w in self._graph_nodes.get(node, {}).get("connections", {}).items() if int(n) in nodes and float(w) > 0}
            adj[node] = conns
        total_seed = sum(max(0.0, float(v)) for v in seeds.values()) or 1.0
        personalization = {n: max(0.0, float(seeds.get(n, 0.0))) / total_seed for n in nodes}
        p = dict(personalization)
        # Adjacency weights do not change between iterations; precompute the
        # outgoing-weight denominator once instead of summing on every visit
        # inside the inner loop. Tiny constant-factor win, but PPR is on the
        # recall hot path.
        denoms = {u: (sum(nbrs.values()) or 1.0) for u, nbrs in adj.items()}
        for _ in range(max(1, iters)):
            newp = {n: alpha * personalization.get(n, 0.0) for n in nodes}
            for u, nbrs in adj.items():
                if not nbrs:
                    continue
                share_per_unit = (1.0 - alpha) * p.get(u, 0.0) / denoms[u]
                for v, w in nbrs.items():
                    newp[v] = newp.get(v, 0.0) + share_per_unit * w
            s = sum(newp.values()) or 1.0
            p = {n: v / s for n, v in newp.items()}
        max_score = max(p.values()) if p else 1.0
        if max_score <= 0:
            return {}
        return {n: v / max_score for n, v in p.items() if n not in seeds or v > 0}

    def _ppr_top_ids_gpu(self, seed_id: int, k: int = 20) -> "list[int]":
        """Like _ppr_scores_gpu, but returns ONLY the top-k node IDs and
        does the top-k on the GPU before any CPU transfer. The full PPR
        score vector for a 1M-edge graph carries hundreds of thousands of
        non-zero floats — most callers (NREM's activated-edges loop) only
        need the highest 20-ish, and don't need labels or content.
        Returning the IDs early collapses ~150KB of GPU→CPU transfer per
        think into ~80 bytes, plus skips the store.get_many SQL round-trip
        downstream callers like think() do for label resolution.

        No longer requires `self._gpu` — _build_gpu_ppr_adjacency picks
        a CUDA device on its own when GpuRecallEngine isn't armed (e.g.
        MM_DB_BACKEND=postgres). PPR is graph-only; embedding tensor not
        needed.
        """
        try:
            import torch
        except ImportError:
            return []
        if (
            getattr(self, "_gpu_ppr_adj", None) is None
            or getattr(self, "_gpu_ppr_dirty", True)
        ):
            if not self._build_gpu_ppr_adjacency():
                return []
        if self._gpu_ppr_adj is None:
            return []

        node_to_idx = self._gpu_ppr_node_to_idx
        idx_to_node = self._gpu_ppr_idx_to_node
        n = len(idx_to_node)
        seed_idx = node_to_idx.get(int(seed_id))
        if seed_idx is None:
            return []

        device = self._gpu_ppr_adj.device
        pers = torch.zeros(n, device=device, dtype=torch.float32)
        pers[seed_idx] = 1.0
        p = pers.clone()
        alpha = self._ppr_alpha
        iters = self._ppr_iters
        for _ in range(max(1, iters)):
            spread = torch.sparse.mm(
                self._gpu_ppr_adj, p.unsqueeze(1)
            ).squeeze(1)
            p = alpha * pers + (1.0 - alpha) * spread
            s = float(p.sum())
            if s > 0:
                p = p / s

        eff_k = min(k + 1, n)
        top = torch.topk(p, eff_k)
        # One sync, transfer ~k ints — vs the full N-vector transfer
        # in _ppr_scores_gpu.
        ind = top.indices.cpu().numpy()
        out: "list[int]" = []
        for i in ind:
            nid = idx_to_node[int(i)]
            if nid == seed_id:
                continue
            out.append(int(nid))
            if len(out) >= k:
                break
        return out

    def think_ids(self, start_id: int, depth: int = 2, k: int = 20) -> "list[int]":
        """Fast think — returns only the top-k activated node IDs.

        Skips label/content resolution (no store.get_many SQL roundtrip)
        and skips the result-dict construction. Used by the NREM hot loop
        where the engine only feeds those IDs into the activated_edges
        set — labels are dead weight there.

        On GPU+CUDA this hits _ppr_top_ids_gpu directly (top-k on GPU,
        ~k ints transferred). Off GPU it falls back to a slim think()
        wrapper that pulls just the IDs."""
        if self._gpu is not None:
            try:
                ids = self._ppr_top_ids_gpu(int(start_id), k=k)
                if ids is not None:
                    return ids
            except Exception:
                pass
        # CPU fallback: reuse think() and strip
        return [int(r.get("id")) for r in self.think(int(start_id), depth=depth) if r.get("id") is not None][:k]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85, engine: Optional[str] = None) -> list[dict]:
        engine = (engine or self._think_engine or "bfs").lower()
        if not self._ensure_node(start_id):
            return []
        if engine == "ppr":
            scores = self._ppr_scores({int(start_id): 1.0}, alpha=self._ppr_alpha, iters=self._ppr_iters, hops=max(1, depth))
            # Batch the label lookup so think(engine='ppr') is one SQL call,
            # not N. Mirrors the BFS-path fix.
            scored_ids = [nid for nid in scores.keys() if nid != start_id]
            mems = self.store.get_many(scored_ids, include_embedding=False) if scored_ids else {}
            results = []
            for node_id in scored_ids:
                mem = mems.get(node_id)
                if mem:
                    results.append({
                        "id": node_id,
                        "label": mem["label"],
                        "activation": round(float(scores[node_id]), 4),
                        "engine": "ppr",
                    })
            results.sort(key=lambda x: -x["activation"])
            return results[:20]

        activation = {start_id: 1.0}
        visited = {start_id}
        queue = [(start_id, 1.0, 0)]
        while queue:
            current, act, level = queue.pop(0)
            if level >= depth or act < 0.01:
                continue
            self._ensure_node(current)
            node = self._graph_nodes.get(current, {})
            for neighbor_id, weight in node.get("connections", {}).items():
                propagated = act * weight * decay
                if propagated < 0.01:
                    continue
                if neighbor_id not in activation or propagated > activation[neighbor_id]:
                    activation[neighbor_id] = propagated
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, propagated, level + 1))
        # Batch resolve labels for all activated nodes in one SQL query
        # instead of N round-trips. With BFS depth 3 over a moderately
        # connected graph this can save 50+ queries.
        active_ids = [nid for nid in activation.keys() if nid != start_id]
        mems = self.store.get_many(active_ids, include_embedding=False) if active_ids else {}
        results = []
        for node_id in active_ids:
            mem = mems.get(node_id)
            if mem:
                results.append({
                    "id": node_id,
                    "label": mem["label"],
                    "activation": round(activation[node_id], 4),
                    "engine": "bfs",
                })
        results.sort(key=lambda x: -x["activation"])
        return results

    def recall_multihop(self, query: str, k: int = 5, hops: int = 2, temporal_weight: float = 0.2) -> list[dict]:
        # Embed once and pass the vector into recall() so the hybrid retrieval
        # path doesn't re-encode the same query. With a real embedding model
        # (FastEmbed e5-large or sentence-transformers on CPU) each encode is
        # 10-50ms — the previous code paid that cost twice per multihop.
        query_emb = self.embedder.embed(query)
        direct = self.recall(
            query, k=k, temporal_weight=temporal_weight, hybrid=True,
            query_vec=query_emb,
        )
        seen = {r["id"] for r in direct}
        all_results = list(direct)

        # First pass: walk every think() expansion, collect the new candidate
        # ids without firing per-id store.get queries. Second pass: one
        # get_many for all of them, then fold in similarity + activation.
        candidates: list[tuple[int, float]] = []  # (id, activation)
        for result in direct:
            for act in self.think(result["id"], depth=hops, engine="ppr"):
                aid = act["id"]
                if aid in seen:
                    continue
                seen.add(aid)
                candidates.append((aid, act["activation"]))
        if candidates:
            cand_ids = [cid for cid, _ in candidates]
            mems = self.store.get_many(cand_ids, include_embedding=True)
            for cid, activation in candidates:
                mem = mems.get(cid)
                if not mem:
                    continue
                emb = mem.get("embedding") or []
                direct_sim = self._cosine_similarity(query_emb, emb) if emb else 0.0
                combined = 0.5 * direct_sim + 0.5 * activation
                all_results.append({
                    "id": cid, "label": mem["label"], "content": mem["content"],
                    "similarity": round(direct_sim, 4), "activation": activation,
                    "combined": round(combined, 4), "relevance": round(combined, 4),
                    "hop": 1, "connections": [],
                })
        all_results.sort(key=lambda x: -x.get("relevance", x.get("combined", x.get("similarity", 0))))
        return all_results[:k * 2]

    def connections(self, mem_id: int, at_time: Optional[float] = None) -> list[dict]:
        conns = self.store.get_connections(mem_id, at_time=at_time)
        if not conns:
            return []
        # Collect every neighbour id and resolve in a single get_many.
        # The previous loop did one SQL round-trip per connection, which on a
        # densely-connected hub memory could mean dozens of synchronous
        # queries just to render labels.
        other_ids = [c["target"] if c["source"] == mem_id else c["source"] for c in conns]
        neighbour_mems = self.store.get_many(other_ids, include_embedding=False)
        results = []
        for c in conns:
            other = c["target"] if c["source"] == mem_id else c["source"]
            mem = neighbour_mems.get(int(other))
            if mem:
                results.append({
                    "id": other,
                    "label": mem["label"],
                    "weight": round(float(c["weight"]), 4),
                    "type": c.get("type", "similar"),
                    "event_time": c.get("event_time"),
                    "valid_from": c.get("valid_from"),
                    "valid_to": c.get("valid_to"),
                })
        return results

    # ------------------------------------------------------------------
    # Dream engine (NREM / REM / Insight) — lazy singleton bound to the
    # same SQLite DB as this store. The v2 mcp wrapper expects
    # `dream(phase=...)` and `dream_stats()` on this class; without them
    # the tools return 500 "internal.error".
    # ------------------------------------------------------------------
    _dream_engine_singleton = None

    def _get_dream_engine(self):
        # Honor MM_DREAM_DISABLED — when an external dream_worker.py owns
        # consolidation we don't want every mazemaker_dream MCP call to
        # also fire a synchronous cycle on the same SQLite DB.
        import os
        if os.getenv("MM_DREAM_DISABLED", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "in-pod dream engine disabled via MM_DREAM_DISABLED — "
                "expected: external dream_worker.py is running"
            )
        if self._dream_engine_singleton is None:
            from dream_engine import DreamEngine  # type: ignore[import]
            self._dream_engine_singleton = DreamEngine.sqlite(
                str(self._db_path), neural_memory=self
            )
        return self._dream_engine_singleton

    def dream(self, phase: str = "all") -> dict:
        """Run a single dream cycle synchronously.

        phase: 'nrem' | 'rem' | 'insight' | 'all' (default).
        Returns the per-phase stats dict.
        """
        # See mazemaker.Memory.dream — same MM_DREAM_DISABLED guard so the
        # MCP wrapper returns a structured "skipped" response instead of
        # surfacing a 500 error to the architect dashboard.
        import os
        if os.getenv("MM_DREAM_DISABLED", "").lower() in ("1", "true", "yes"):
            return {
                "ok": True,
                "phase": phase,
                "status": "skipped",
                "note": (
                    "in-pod dream engine disabled — external dream_worker.py "
                    "owns consolidation. Cycles still land in dream_sessions; "
                    "watch mazemaker_dream_stats."
                ),
            }
        eng = self._get_dream_engine()
        phase = (phase or "all").strip().lower()
        if phase == "all":
            return eng.dream_now()
        if phase == "nrem":
            return {"nrem": eng._phase_nrem()}
        if phase == "rem":
            return {"rem": eng._phase_rem()}
        if phase == "insight":
            return {"insights": eng._phase_insights()}
        raise ValueError(f"unknown dream phase: {phase}")

    def dream_stats(self) -> dict:
        """Aggregate dream-engine stats from the backend.

        Read-only; bypasses the _get_dream_engine guard so dream_stats
        keeps working when MM_DREAM_DISABLED is set (external dream_worker
        is still writing rows that this method reads).

        Backend resolution mirrors __init__: when MM_DB_BACKEND=postgres
        (the canonical setup for the Pro/Enterprise pod where the
        dream_worker container writes dream_sessions to pgvector), we
        read the Postgres store. Otherwise the SQLite primary. Without
        this branch, an in-pod mcp with MM_DREAM_DISABLED=1 would
        silently fall back to a stale SQLite dream_sessions table while
        the external dream_worker keeps appending to Postgres.
        """
        if self._dream_engine_singleton is not None:
            return self._dream_engine_singleton._backend.get_dream_stats()
        backend_choice = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
        if backend_choice == "postgres" and has_feature("postgres"):
            from dream_postgres_store import DreamPostgresStore  # type: ignore[import]
            return DreamPostgresStore().get_dream_stats()
        from dream_engine import SQLiteDreamBackend  # type: ignore[import]
        return SQLiteDreamBackend(str(self._db_path)).get_dream_stats()

    def graph(self) -> dict:
        stats = self.store.get_stats()
        edges = []
        seen = set()
        # In lazy mode graph summary is still truthful: read from the
        # store's edge table directly, not only hydrated nodes.
        rows = self.store.top_weighted_edges(limit=500)
        for r in rows:
            src = int(r["source"])
            tgt = int(r["target"])
            etype = r.get("type") or r.get("edge_type") or "similar"
            key = tuple(sorted([src, tgt])) + (etype,)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": src, "to": tgt,
                          "weight": round(float(r["weight"]), 3), "type": etype})
        return {"nodes": stats["memories"], "edges": stats["connections"], "top_edges": edges}

    def stats(self) -> dict:
        graph = self.store.stats() if hasattr(self.store, "stats") else self.store.get_stats()
        return {
            "memories": graph["memories"],
            "connections": graph["connections"],
            "revisions": graph.get("revisions", 0),
            "embedding_dim": self.dim,
            "embedding_backend": self.embedder.backend.__class__.__name__,
            "embed_fingerprint": self._embed_fingerprint,
            "dim_locked": self._dim_locked,
            "dim_mismatch_reason": self._dim_mismatch_reason or None,
            "quarantined_dim": self._quarantined_dim,
            "retrieval_mode": self._retrieval_mode,
            "lazy_graph": self._lazy_graph,
            "hnsw_enabled": self._hnsw_enabled,
        }

    def prune_connections_below(self, threshold: float) -> int:
        """Drop all graph edges with weight below `threshold`. Returns count deleted.

        Designed for one-shot graph hygiene on saturated stores — the auto_connect
        bug at sim>0.45 (now 0.70) accumulated ~390 connections per memory; this
        sweeps the existing hairball without touching memories themselves.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        deleted = self.store.prune_connections_below(float(threshold))
        # Mirror the DELETE in the in-memory graph cache. Filter out sub-threshold
        # edges per node — DO NOT clear the whole connections dict, or every
        # surviving edge (weight >= threshold) disappears from recall/think
        # traversals until the cache is reloaded.
        thr = float(threshold)
        for node in self._graph_nodes.values():
            conns = node.get("connections")
            if not conns:
                continue
            node["connections"] = {nid: w for nid, w in conns.items() if w >= thr}
        self._hnsw_dirty = True
        return int(deleted)

    def close(self):
        if self._cpp:
            try:
                self._cpp.shutdown()
            except Exception:
                pass
            self._cpp = None
        self.store.close()

    try:
        from fast_ops import cosine_similarity as _cosine_sim_fast
    except ImportError:
        _cosine_sim_fast = None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity with strict dim-equality.

        Treats dim-mismatched vectors as "no signal" (returns 0.0) instead of
        silently truncating via zip. A mismatch means the two embeddings came
        from different models — comparing them across dim is meaningless and
        produces noise that pollutes recall.
        """
        if not a or not b:
            return 0.0
        # Strict dim guard: zip-based fallback would otherwise truncate to the
        # shorter vector and return a meaningless partial dot product.
        if len(a) != len(b):
            return 0.0
        if Mazemaker._cosine_sim_fast is not None:
            import numpy as np
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b, dtype=np.float64)
            # numpy/fast_ops would also fail loudly on mismatch, but the
            # explicit early return above keeps the contract uniform across
            # both code paths.
            return float(Mazemaker._cosine_sim_fast(a, b))
        dot = sum(x * y for x, y in zip(a, b))
        na = (sum(x * x for x in a)) ** 0.5
        nb = (sum(x * x for x in b)) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
