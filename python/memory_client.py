#!/usr/bin/env python3
"""memory_client.py - Python client for Neural Memory Adapter.

This is the hot path: SQLite persistence, optional C++/GPU/HNSW indexes,
hybrid retrieval, typed temporal graph edges, and PPR thinking.
"""
from __future__ import annotations

import ctypes
import math
import os
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Find the shared library
# ============================================================================

def _find_lib():
    candidates = [
        Path(__file__).parent.parent / "build" / "libneural_memory.so",
        Path.home() / "projects" / "neural-memory-adapter" / "build" / "libneural_memory.so",
        Path("/usr/local/lib/libneural_memory.so"),
        Path("/usr/lib/libneural_memory.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("libneural_memory.so not found. Build first: cd build && cmake --build .")


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

DB_PATH = Path.home() / ".neural_memory" / "memory.db"


def _resolve_hf_snapshot(model_name: str) -> Optional[str]:
    """Return a local snapshot directory for model_name, or None if not cached.

    Checks ~/.neural_memory/models first, then the default HF hub cache.
    Passing a snapshot path directly to SentenceTransformer / CrossEncoder
    avoids any network contact with the HF Hub.
    """
    safe_name = model_name.replace("/", "--")
    search_dirs = [
        Path.home() / ".neural_memory" / "models",
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
        self.conn.executescript(SCHEMA)
        self._ensure_schema_extensions()
        self._fts_available = self._ensure_fts()
        self.conn.commit()
        self._lock = threading.Lock()
        self._checkpoint_thread = threading.Thread(target=self._bg_checkpoint, daemon=True)
        self._checkpoint_thread.start()

    def get_meta(self, key: str) -> Optional[str]:
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

    def _ensure_fts(self) -> bool:
        try:
            self.conn.executescript(FTS_SCHEMA)
            mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            fts_count = self.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            if mem_count and fts_count == 0:
                self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            return True
        except sqlite3.OperationalError:
            return False

    def _connect_reader(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _bg_checkpoint(self):
        while True:
            time.sleep(60)
            try:
                with self._lock:
                    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            except Exception:
                pass

    @staticmethod
    def _unpack_embedding(blob: bytes | None) -> list[float]:
        if not blob:
            return []
        dim = len(blob) // 4
        return list(struct.unpack(f"{dim}f", blob))

    @staticmethod
    def _sanitize_fts_query(query: str, mode: str = "and") -> str:
        tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]{1,}", query or "")
        tokens = [t.replace('"', '') for t in tokens[:12]]
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
        placeholders = ",".join("?" for _ in ids)
        cols = "id, label, content, salience, created_at, last_accessed, access_count"
        if include_embedding:
            cols += ", embedding"
        rows = self.conn.execute(f"SELECT {cols} FROM memories WHERE id IN ({placeholders})", tuple(ids)).fetchall()
        out: dict[int, dict] = {}
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
            decayed = old_salience * math.exp(-0.03 * idle_days)
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
        weight = max(0.0, min(1.0, float(weight)))
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
        fts_query = self._sanitize_fts_query(query, mode="and")
        if self._fts_available and fts_query:
            try:
                conn = self._connect_reader()
                try:
                    rows = conn.execute(
                        "SELECT rowid AS id, bm25(memories_fts) AS rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                    return [{"id": int(r["id"]), "score": 1.0 / (i + 1), "rank": float(r["rank"]), "channel": "bm25"} for i, r in enumerate(rows)]
                finally:
                    conn.close()
            except sqlite3.OperationalError:
                pass
        # Fallback lexical overlap.
        q = set(re.findall(r"\w+", (query or "").lower()))
        if not q:
            return []
        scored = []
        for m in self.get_all():
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
        scored = []
        lowered = [e.lower() for e in entities]
        for m in self.get_all():
            text = ((m.get("label") or "") + " " + (m.get("content") or "")).lower()
            hits = [e for e in lowered if e in text]
            if hits:
                scored.append({"id": m["id"], "score": len(hits) / len(lowered), "matched_entities": hits, "channel": "entity"})
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def search_temporal(self, query: str, limit: int = 50, now: Optional[float] = None) -> list[dict]:
        now = now or time.time()
        q = (query or "").lower()
        where = ""
        params: list[Any] = []
        if any(w in q for w in ("today", "heute")):
            where = "WHERE created_at >= ?"
            params.append(now - 86400)
        elif any(w in q for w in ("yesterday", "gestern")):
            where = "WHERE created_at BETWEEN ? AND ?"
            params.extend([now - 2 * 86400, now - 86400])
        elif "last week" in q or "letzte woche" in q:
            where = "WHERE created_at >= ?"
            params.append(now - 7 * 86400)
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

    def close(self):
        self.conn.close()


# ============================================================================
# Neural Memory Client
# ============================================================================

class NeuralMemory:
    """Python interface to the Neural Memory system."""

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        embedding_backend: str = "auto",
        use_mssql: bool = False,
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
    ):
        if embedder is not None:
            self.embedder = embedder
        else:
            from embed_provider import EmbeddingProvider
            self.embedder = EmbeddingProvider(backend=embedding_backend)

        if use_mssql:
            from mssql_store import MSSQLStore
            self.store = MSSQLStore()
        else:
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
        self._recall_score_floor = max(0.0, float(recall_score_floor or 0.0))
        self._channel_weights = {
            "semantic": 1.0,
            "bm25": 0.9,
            "entity": 1.0,
            "temporal": 0.35,
            "ppr": 0.55,
            "salience": 0.25,
        }
        if isinstance(channel_weights, dict):
            self._channel_weights.update({k: float(v) for k, v in channel_weights.items() if v is not None})

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

        self._cpp = None
        self._cpp_id_map: dict[int, int] = {}
        if use_cpp:
            try:
                from cpp_bridge import NeuralMemoryCpp
                self._cpp = NeuralMemoryCpp()
                self._cpp.initialize(dim=self.dim)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("C++ bridge unavailable, falling back to Python: %s", e)
                self._cpp = None

        self._gpu = None
        if Path(db_path) == DB_PATH:
            try:
                from gpu_recall import GpuRecallEngine
                self._gpu = GpuRecallEngine()
                if not self._gpu.load(embed_fn=self.embedder.embed):
                    self._gpu = None
            except Exception:
                self._gpu = None

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
        """
        if not hasattr(self.store, "get_meta"):
            # MSSQLStore doesn't expose db_meta; treat as unlocked.
            return False, ""
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
            # MSSQL store path (no get_all_connections) — fall back to the
            # per-memory query so behaviour is preserved on non-SQLite backends.
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
        cache_hit = (
            at_time is None
            and cached is not None
            and not cached.get("label", "").startswith("memory:")
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

    def _ensure_hnsw(self) -> bool:
        """Build or incrementally update the HNSW ANN index.

        First call constructs from scratch (paying O(N log N) once).
        Subsequent calls after writes only add the new memory ids via
        hnswlib.add_items, which is O(M log N) for M new items — way
        cheaper than the previous always-rebuild path that paid O(N log N)
        on every recall after even a single write.

        Falls back to full rebuild only when the live capacity is exhausted
        or when the index contains ids the store no longer has (deletes).
        """
        if self._hnsw_enabled is False or str(self._hnsw_enabled).lower() in {"0", "false", "no", "off"}:
            return False
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
        return True

    # -- write path -----------------------------------------------------------

    def remember(self, text: str, label: str = "", detect_conflicts: bool = True, auto_connect: bool = True) -> int:
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
                # the previous \`if other_emb and cos(...) < 0.85: continue\`
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
                    self.store.add_revision(int(other["id"]), old_content, text, "conflict_fusion")
                    self.store.update_memory(int(other["id"]), fused, embedding, label=label)
                    # Cache MUST mirror what was actually written to DB.
                    # Was setting content=text here while DB held \`fused\`,
                    # so recall via _graph_nodes saw the raw new content but
                    # recall via store.get_many saw the [CANONICAL]/[PREVIOUSLY]
                    # marker — same memory, two contents, depending on path.
                    self._graph_nodes[int(other["id"])] = {
                        "embedding": embedding, "label": label,
                        "content": fused, "connections": {},
                    }
                    self._refresh_connections(int(other["id"]))
                    self._hnsw_dirty = True
                    return int(other["id"])

        mem_id = self.store.store(label, text, embedding)
        self._graph_nodes[mem_id] = {"embedding": embedding, "label": label, "content": text, "connections": {}}

        if self._cpp:
            try:
                cpp_id = self._cpp.store(embedding, label, text)
                self._cpp_id_map[cpp_id] = mem_id
            except Exception:
                pass
        self._hnsw_dirty = True

        if auto_connect:
            self._auto_connect(mem_id, embedding, text)

        return mem_id

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

        for other_id, sim in candidates:
            other_node = self._graph_nodes.get(other_id, {})
            edge_type = self._infer_edge_type(text, other_node.get("content", ""))
            self.store.add_connection(mem_id, other_id, sim, edge_type=edge_type)
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
            row = self.store.conn.execute("SELECT created_at FROM memories WHERE id = ?", (mem_id,)).fetchone()
            if row and row[0]:
                return self._compute_temporal_score_from_mem({"created_at": row[0]}, now)
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
        # HNSW ANN path when enabled and worthwhile.
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

        if self._gpu:
            try:
                gpu_results = self._gpu.recall(query, k=limit)
                if gpu_results:
                    return [{"id": int(r["id"]), "score": float(r.get("similarity", 0.0)), "similarity": float(r.get("similarity", 0.0)), "channel": "semantic"} for r in gpu_results]
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

    def _parallel_retrieve(self, query: str, query_vec: list[float], limit: int, now: float) -> dict[str, list[dict]]:
        # Kept intentionally stdlib. SQLite reads are cheap and each lexical method
        # opens its own reader, so this can be parallelized later without API churn.
        return {
            "semantic": self._semantic_candidates(query, query_vec, limit),
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
    ) -> list[dict]:
        if k <= 0:
            return []
        query_vec = query_vec or self.embedder.embed(query)
        now = time.time()
        limit = max(k * 4, self._retrieval_candidates)
        if hybrid is None:
            hybrid = self._retrieval_mode in {"hybrid", "advanced", "skynet"}

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

        mems = self.store.get_many(list(fused.keys()), include_embedding=True)

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
                "combined": round(relevance, 6),
                "relevance": round(relevance, 6),
                "channel_scores": {ch: round(float(v), 6) for ch, v in data.get("channel_scores", {}).items()},
                "connections": connected,
                "created_at": mem.get("created_at"),
                "last_accessed": mem.get("last_accessed"),
                "access_count": mem.get("access_count", 0),
            })

        results.sort(key=lambda x: -x["relevance"])
        use_rerank = self._rerank if rerank is None else bool(rerank)
        if use_rerank:
            results = self._rerank_results(query, results, k)

        # Hard noise floor — drop sub-floor results before MMR/truncation so
        # garbage queries return [] instead of weak top-k that operators
        # mistake for signal.
        floor = self._recall_score_floor if score_floor is None else float(score_floor or 0.0)
        if floor > 0.0:
            results = [r for r in results if r.get("relevance", 0.0) >= floor]

        # MMR diversification: trade marginal relevance for non-redundancy in
        # the returned set. Greedy O(k * |candidates|), bounded by k * limit.
        # Skipped when mmr_lambda is at its default 0.0 (pure relevance) or
        # the candidate pool already has <2 items.
        eff_lambda = self._mmr_lambda if mmr_lambda is None else max(0.0, min(1.0, float(mmr_lambda or 0.0)))
        if eff_lambda > 0.0 and len(results) > 1 and k > 1:
            results = self._mmr_rerank(results, mems, eff_lambda, k)

        final = results[:k]
        if touch:
            for r in final:
                try:
                    self.store.touch(r["id"])
                except Exception:
                    pass
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

    def _ppr_scores(self, seeds: dict[int, float], alpha: float = 0.15, iters: int = 20, hops: int = 2) -> dict[int, float]:
        if not seeds:
            return {}
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
        direct = self.recall(query, k=k, temporal_weight=temporal_weight, hybrid=True)
        seen = {r["id"] for r in direct}
        all_results = list(direct)
        query_emb = self.embedder.embed(query)

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

    def graph(self) -> dict:
        stats = self.store.get_stats()
        edges = []
        seen = set()
        # In lazy mode graph summary is still truthful: read from SQLite, not only hydrated nodes.
        rows = self.store.conn.execute("SELECT source_id, target_id, weight, edge_type FROM connections ORDER BY weight DESC LIMIT 500").fetchall()
        for r in rows:
            key = tuple(sorted([r["source_id"], r["target_id"]])) + (r["edge_type"] or "similar",)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": r["source_id"], "to": r["target_id"], "weight": round(float(r["weight"]), 3), "type": r["edge_type"] or "similar"})
        return {"nodes": stats["memories"], "edges": stats["connections"], "top_edges": edges[:10]}

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
        cur = self.store.conn.execute(
            "DELETE FROM connections WHERE weight < ?", (float(threshold),)
        )
        deleted = cur.rowcount
        self.store.conn.commit()
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
        if NeuralMemory._cosine_sim_fast is not None:
            import numpy as np
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b, dtype=np.float64)
            # numpy/fast_ops would also fail loudly on mismatch, but the
            # explicit early return above keeps the contract uniform across
            # both code paths.
            return float(NeuralMemory._cosine_sim_fast(a, b))
        dot = sum(x * y for x, y in zip(a, b))
        na = (sum(x * x for x in a)) ** 0.5
        nb = (sum(x * x for x in b)) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
