#!/usr/bin/env python3
"""
memory_client.py - Python client for Neural Memory Adapter
Wraps the C++ library via ctypes. Uses embed_provider for text->vector.
"""

import ctypes
import json
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

from classify_memory_kind import classify_memory_kind
from entity_extraction import EntityRegistry
from schema_upgrade import SchemaUpgrade

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
        ("content", ctypes.c_char * 1024),
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

# Salience decay constants — see _effective_salience()
SALIENCE_AGE_DECAY_K = 0.001      # per day; gentle background forgetting
SALIENCE_ACCESS_BOOST = 0.05      # per log access
SALIENCE_MIN = 0.1
SALIENCE_MAX = 2.0

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
    -- Bi-temporal fields (Graphiti-style). All nullable for backwards compat:
    -- pre-existing edges have NULL for these and are treated as always-valid.
    event_time REAL DEFAULT NULL,         -- when the claimed fact occurred
    ingestion_time REAL DEFAULT NULL,     -- when the system learned it (defaults to created_at if NULL)
    valid_from REAL DEFAULT NULL,         -- edge validity start (NULL = no lower bound)
    valid_to REAL DEFAULT NULL,           -- edge validity end (NULL = still valid)
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_id);
CREATE INDEX IF NOT EXISTS idx_connections_edge_type_source ON connections(edge_type, source_id);

-- H19 Active Contradiction Replacement: archive table for superseded memories.
-- When supersede fires, the OLD content moves here and the original memories
-- row is replaced cleanly. Audit history preserved; current state stays clean.
CREATE TABLE IF NOT EXISTS superseded_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_id INTEGER NOT NULL,
    content TEXT,
    label TEXT,
    embedding BLOB,
    salience REAL,
    superseded_by INTEGER,
    superseded_at REAL DEFAULT (unixepoch()),
    superseded_reason TEXT,
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_superseded_original ON superseded_memories(original_id);
CREATE INDEX IF NOT EXISTS idx_superseded_by ON superseded_memories(superseded_by);
"""

# Columns to add via ALTER TABLE on existing DBs (migration for bi-temporal fields).
_BITEMPORAL_COLUMNS = [
    ("event_time", "REAL DEFAULT NULL"),
    ("ingestion_time", "REAL DEFAULT NULL"),
    ("valid_from", "REAL DEFAULT NULL"),
    ("valid_to", "REAL DEFAULT NULL"),
]


def _migrate_bitemporal(conn: sqlite3.Connection) -> None:
    """Add bi-temporal columns to existing connections tables. Idempotent."""
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(connections)")}
    except sqlite3.DatabaseError:
        return
    for name, decl in _BITEMPORAL_COLUMNS:
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE connections ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError:
                # Race with concurrent migration or unsupported on this sqlite; skip silently.
                pass
    conn.commit()

class SQLiteStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        self.conn.execute("PRAGMA synchronous=NORMAL") # Faster writes
        self.conn.executescript(SCHEMA)
        _migrate_bitemporal(self.conn)
        self.conn.commit()
        # Phase 7: extend with typed/temporal/governance/locus columns. Idempotent.
        SchemaUpgrade(str(db_path)).upgrade()
        self._lock = threading.Lock()

    def store(
        self,
        label: str,
        content: str,
        embedding: list[float],
        *,
        kind: Optional[str] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        origin_system: Optional[str] = None,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        transaction_time: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        salience: Optional[float] = None,
        procedural_score: Optional[float] = None,
    ) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        if transaction_time is None:
            transaction_time = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        # Build INSERT dynamically: only include typed columns when caller
        # provided a non-None value. Schema-defined defaults handle the rest.
        cols = ["label", "content", "embedding"]
        vals: list[Any] = [label, content, blob]
        for col_name, col_value in (
            ("kind", kind),
            ("confidence", confidence),
            ("source", source),
            ("origin_system", origin_system),
            ("valid_from", valid_from),
            ("valid_to", valid_to),
            ("transaction_time", transaction_time),
            ("metadata_json", metadata_json),
            ("salience", salience),
            ("procedural_score", procedural_score),
        ):
            if col_value is not None:
                cols.append(col_name)
                vals.append(col_value)

        placeholders = ",".join("?" * len(vals))
        col_list = ",".join(cols)

        with self._lock:
            cur = self.conn.execute(
                f"INSERT INTO memories ({col_list}) VALUES ({placeholders})",
                tuple(vals),
            )
            new_id = cur.lastrowid
            # Phase 7 Commit 5: sync FTS5 index for sparse retrieval. Silent
            # no-op if memories_fts wasn't created (SQLite without FTS5 support).
            # Skip kind='entity' rows — entities are derived nodes, not user
            # memories; indexing their "Entity: X" content adds sparse-search
            # noise without value. Caught by phase7_audit.py FTS5 sync delta.
            if kind != "entity":
                try:
                    self.conn.execute(
                        "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                        (new_id, content),
                    )
                except sqlite3.OperationalError:
                    pass
            self.conn.commit()
            return new_id
    
    def get_all(self) -> list[dict]:
        import struct
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, label, content, embedding, salience, access_count, created_at FROM memories ORDER BY id"
            ).fetchall()
        results = []
        for row in rows:
            id_, label, content, blob, salience, access_count, created_at = row
            if blob is None:
                continue
            dim = len(blob) // 4
            embedding = list(struct.unpack(f'{dim}f', blob))
            results.append({
                'id': id_, 'label': label, 'content': content,
                'embedding': embedding, 'salience': salience,
                'access_count': access_count, 'created_at': created_at,
            })
        return results

    def get_meta_many(self, ids: list[int]) -> dict[int, dict]:
        """Batch-fetch salience/access/created_at for a set of ids.

        Used by the C++ fast-path in recall() so salience can be applied
        without reloading the full embedding blob.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, salience, access_count, created_at FROM memories WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {
            r[0]: {'salience': r[1], 'access_count': r[2], 'created_at': r[3]}
            for r in rows
        }
    
    def get(self, id_: int) -> Optional[dict]:
        import struct
        row = self.conn.execute(
            "SELECT id, label, content, embedding, salience, access_count FROM memories WHERE id = ?",
            (id_,)
        ).fetchone()
        if not row:
            return None
        id_, label, content, blob, salience, access_count = row
        dim = len(blob) // 4
        embedding = list(struct.unpack(f'{dim}f', blob))
        return {
            'id': id_, 'label': label, 'content': content,
            'embedding': embedding, 'salience': salience,
            'access_count': access_count
        }
    
    def touch(self, id_: int):
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET last_accessed = unixepoch(), access_count = access_count + 1 WHERE id = ?",
                (id_,)
            )
            self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float,
                       edge_type: str = "similar",
                       event_time: Optional[float] = None,
                       ingestion_time: Optional[float] = None,
                       valid_from: Optional[float] = None,
                       valid_to: Optional[float] = None):
        """Insert or replace an edge with optional bi-temporal metadata.

        All temporal fields are optional; omitting them preserves pre-bitemporal
        behavior (edge is always-valid). `ingestion_time` defaults to now when
        any temporal field is set, so edges carrying event_time get a meaningful
        ingestion stamp without requiring callers to supply it explicitly.
        """
        import time as _time
        if ingestion_time is None and (event_time is not None or valid_from is not None or valid_to is not None):
            ingestion_time = _time.time()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO connections
                   (source_id, target_id, weight, edge_type, event_time, ingestion_time, valid_from, valid_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (source, target, weight, edge_type, event_time, ingestion_time, valid_from, valid_to),
            )
            self.conn.commit()

    def get_connections(self, node_id: int, at_time: Optional[float] = None,
                        include_expired: bool = False) -> list[dict]:
        """Return edges touching node_id.

        - Default (no args): returns currently-valid edges only (edges with
          `valid_to IS NULL` or `valid_to > now`). This keeps dream-pruned
          soft-deleted edges out of default recall.
        - `at_time=ts`: filter to edges valid at that instant:
          `valid_from <= at_time <= valid_to` (NULLs treated as unbounded).
        - `include_expired=True`: return ALL edges regardless of validity
          (useful for audit / replay over `connection_history`).
        """
        import time as _time
        rows = self.conn.execute(
            """SELECT source_id, target_id, weight, edge_type,
                      event_time, ingestion_time, valid_from, valid_to
               FROM connections
               WHERE source_id = ? OR target_id = ? ORDER BY weight DESC""",
            (node_id, node_id),
        ).fetchall()
        now = at_time if at_time is not None else _time.time()
        out = []
        for r in rows:
            vf, vt = r[6], r[7]
            if not include_expired:
                # Filter expired edges (H6 soft-delete awareness)
                if vf is not None and now < vf:
                    continue
                if vt is not None and now > vt:
                    continue
            out.append({
                'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3],
                'event_time': r[4], 'ingestion_time': r[5],
                'valid_from': vf, 'valid_to': vt,
            })
        return out

    def set_edges_valid_to(self, node_id: int, ts: float,
                           edge_type: Optional[str] = None) -> int:
        """H3/H6: mark all currently-valid edges touching node_id as expired at ts.

        If `edge_type` is provided, only edges of that type are affected.
        Returns count of rows updated.

        Used by:
          - remember() supersede branch (H3) — invalidate edges on conflict
          - dream_engine NREM prune (H6) — soft-delete instead of hard-delete
        """
        with self._lock:
            if edge_type is None:
                cur = self.conn.execute(
                    """UPDATE connections SET valid_to = ?
                       WHERE (source_id = ? OR target_id = ?)
                         AND valid_to IS NULL""",
                    (ts, node_id, node_id),
                )
            else:
                cur = self.conn.execute(
                    """UPDATE connections SET valid_to = ?
                       WHERE (source_id = ? OR target_id = ?)
                         AND edge_type = ?
                         AND valid_to IS NULL""",
                    (ts, node_id, node_id, edge_type),
                )
            self.conn.commit()
            return cur.rowcount

    def archive_superseded(self, original_id: int, content: str, label: str,
                           embedding: list[float], salience: float,
                           superseded_by: Optional[int] = None,
                           superseded_at: Optional[float] = None,
                           superseded_reason: str = "") -> int:
        """H19: archive a memory's prior content to superseded_memories before
        replace_memory() rewrites the row. Returns the audit-table row id.
        """
        import time as _time
        if superseded_at is None:
            superseded_at = _time.time()
        blob = struct.pack(f'{len(embedding)}f', *embedding) if embedding else None
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO superseded_memories
                   (original_id, content, label, embedding, salience,
                    superseded_by, superseded_at, superseded_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (original_id, content, label, blob, salience,
                 superseded_by, superseded_at, superseded_reason),
            )
            self.conn.commit()
            return cur.lastrowid

    def replace_memory(self, memory_id: int, content: str, label: str,
                       embedding: list[float],
                       *,
                       kind: Optional[str] = None,
                       confidence: Optional[float] = None,
                       source: Optional[str] = None,
                       origin_system: Optional[str] = None,
                       valid_from: Optional[float] = None,
                       valid_to: Optional[float] = None,
                       transaction_time: Optional[float] = None,
                       metadata: Optional[dict[str, Any]] = None) -> None:
        """H19: cleanly replace a memory's content + label + embedding in place.
        No `[SUPERSEDED]` prefix. Caller should have called archive_superseded()
        first to preserve audit history.

        Phase 7 Commit 11 (carryover from C2/C3 punt): typed kwargs now flow
        through. When supersession fires, the new memory's kind/confidence/
        source/origin_system/temporal/metadata can replace the old typing.
        Defaults preserve old typing when caller omits kwargs.
        """
        blob = struct.pack(f'{len(embedding)}f', *embedding) if embedding else None

        # Build dynamic UPDATE: always update content/label/embedding;
        # additionally update typed cols when caller supplied them.
        sets: list[str] = ["content = ?", "label = ?", "embedding = ?"]
        vals: list[Any] = [content, label, blob]
        metadata_json = json.dumps(metadata) if metadata else None

        for col_name, col_value in (
            ("kind", kind),
            ("confidence", confidence),
            ("source", source),
            ("origin_system", origin_system),
            ("valid_from", valid_from),
            ("valid_to", valid_to),
            ("transaction_time", transaction_time),
            ("metadata_json", metadata_json),
        ):
            if col_value is not None:
                sets.append(f"{col_name} = ?")
                vals.append(col_value)
        vals.append(memory_id)

        with self._lock:
            self.conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                tuple(vals),
            )
            # Phase 7 Commit 5: refresh FTS5 row so sparse_search doesn't return
            # stale superseded content. Silent no-op if FTS5 unavailable.
            try:
                self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
                self.conn.execute(
                    "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                    (memory_id, content),
                )
            except sqlite3.OperationalError:
                pass
            self.conn.commit()

    def set_edge_valid_to(self, source_id: int, target_id: int, ts: float) -> int:
        """H6: mark a specific edge (source<->target, either direction) as expired at ts.

        Returns count updated (0, 1, or 2 if duplicate rows exist in both directions).
        """
        with self._lock:
            cur = self.conn.execute(
                """UPDATE connections SET valid_to = ?
                   WHERE ((source_id = ? AND target_id = ?)
                       OR (source_id = ? AND target_id = ?))
                     AND valid_to IS NULL""",
                (ts, source_id, target_id, target_id, source_id),
            )
            self.conn.commit()
            return cur.rowcount
    
    def get_stats(self) -> dict:
        # Phase 7 Commit 3: exclude kind='entity' rows from the user-facing
        # memory count. Entities are derived nodes in the unified graph; the
        # historic 'memories' count means "memories the user added", not
        # "all rows in the memories table". Surface entity count separately.
        mem_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind IS NULL OR kind != 'entity'"
        ).fetchone()[0]
        entity_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind = 'entity'"
        ).fetchone()[0]
        conn_count = self.conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        return {'memories': mem_count, 'connections': conn_count, 'entities': entity_count}
    
    def close(self):
        self.conn.close()


# ============================================================================
# Neural Memory Client
# ============================================================================

class NeuralMemory:
    """
    Python interface to the Neural Memory system.
    
    Usage:
        mem = NeuralMemory()
        mem.remember("The user has a dog named Lou")
        mem.remember("Working on BTQuant trading platform")
        results = mem.recall("What pet does the user have?")
    """
    
    def __init__(self, db_path: str | Path = DB_PATH, embedding_backend: str = "auto",
                 use_mssql: bool = False, use_cpp: bool = True,
                 rerank: bool = False,
                 rerank_model: Optional[str] = None,
                 use_hnsw: bool = True,
                 lazy_graph: bool = False,
                 hnsw_ef_construction: int = 200,
                 hnsw_m: int = 16,
                 hnsw_ef: int = 100,
                 salience_multiply: bool = True,
                 hnsw_index_path: Optional[str] = None,
                 hnsw_save_every: int = 50):
        from embed_provider import EmbeddingProvider

        self.embedder = EmbeddingProvider(backend=embedding_backend)

        if use_mssql:
            from mssql_store import MSSQLStore
            self.store = MSSQLStore()
        else:
            self.store = SQLiteStore(db_path)

        # Phase 7 Commit 3: entity registry (kind='entity' nodes + mentions_entity edges).
        # Skipped for MSSQL backend (Commit 3 punt; MSSQL lacks _lock attr).
        self.entities = EntityRegistry(self.store) if not use_mssql else None

        self.dim = self.embedder.dim

        # Cross-encoder reranker (opt-in, lazy-loaded).
        # Keeps default behavior identical when rerank=False.
        # Caught 2026-05-01 in AE-domain bench: English-trained MiniLM
        # MUST_marco model regressed Spanish queries (R@5 0.33 → 0.0).
        # Resolution chain for the model name:
        #   1. constructor kwarg `rerank_model` (highest)
        #   2. NM_RERANK_MODEL env var
        #   3. multilingual default (handles AE Spanish + English)
        # Original English-only default kept available via env override.
        import os
        if rerank_model is None:
            rerank_model = os.environ.get(
                "NM_RERANK_MODEL",
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            )
        self._rerank = bool(rerank)
        self._rerank_model_name = rerank_model
        self._rerank_model = None  # lazy

        # H5: Salience opt-out. Default True preserves shipped Phase B behavior
        # (Bucket-C default shift). Set False to get pre-Phase-B recall ordering.
        self._salience_multiply = bool(salience_multiply)

        # H1: Remember ef for observability
        self._hnsw_ef = hnsw_ef

        # H4: HNSW persistence — index saved to disk next to the DB.
        # Auto-derives from db_path if not explicit.
        if hnsw_index_path is None:
            try:
                hnsw_index_path = str(Path(db_path).with_suffix(".hnsw.bin"))
            except Exception:
                hnsw_index_path = str(Path.home() / ".neural_memory" / "hnsw.bin")
        self._hnsw_index_path = hnsw_index_path
        self._hnsw_save_every = max(1, int(hnsw_save_every))
        self._hnsw_writes_since_save = 0

        # C++ SIMD index for fast retrieval (primary search path)
        self._cpp = None
        if use_cpp:
            try:
                from cpp_bridge import NeuralMemoryCpp
                self._cpp = NeuralMemoryCpp()
                self._cpp.initialize(dim=self.dim)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "C++ bridge unavailable, falling back to Python: %s", e
                )
                self._cpp = None

        # HNSW index (optional, Python-only fast retrieval when C++ unavailable).
        # Safe to coexist with C++: we prefer C++ in recall() but HNSW fills in
        # if the C++ bridge fails. The index is in-memory and reloads from store.
        self._hnsw = None
        self._hnsw_capacity = 0
        self._hnsw_count = 0
        if use_hnsw and self._cpp is None:
            try:
                import hnswlib  # type: ignore
                self._hnsw = hnswlib.Index(space='cosine', dim=self.dim)
                # Initial capacity grows on demand via resize_index
                self._hnsw_capacity = 1024
                self._hnsw.init_index(max_elements=self._hnsw_capacity,
                                      ef_construction=hnsw_ef_construction, M=hnsw_m)
                # H1: expose ef as tunable param (was hardcoded)
                self._hnsw.set_ef(hnsw_ef)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("hnswlib unavailable: %s", exc)
                self._hnsw = None

        # In-memory graph for spreading activation.
        # lazy_graph=True skips the eager full-DB load; nodes are fetched on
        # demand via _ensure_node(). Useful past ~10k memories where eager load
        # becomes RAM-bound.
        self._graph_nodes: dict[int, dict] = {}
        self._lazy_graph = bool(lazy_graph)
        if not self._lazy_graph:
            self._load_from_store()
        else:
            # Still build the C++/HNSW side-indexes, just without populating
            # _graph_nodes. Connections load on demand in _ensure_node().
            self._cpp_id_map = {}
            self._load_indexes_only()

    def _load_from_store(self):
        """Load existing memories into in-memory graph + C++/HNSW indexes."""
        all_mems = self.store.get_all()
        for mem in all_mems:
            self._graph_nodes[mem['id']] = {
                'embedding': mem['embedding'],
                'label': mem['label'],
                'connections': {}
            }

        # Load connections
        for mem in all_mems:
            conns = self.store.get_connections(mem['id'])
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                self._graph_nodes[mem['id']]['connections'][other] = c['weight']

        # Load into C++ SIMD index + build ID mapping
        self._cpp_id_map = {}  # cpp_id -> sqlite_id
        if self._cpp:
            for mem in all_mems:
                emb = mem.get('embedding', [])
                if emb and len(emb) == self.dim:
                    cpp_id = self._cpp.store(emb, mem.get('label', ''), mem.get('content', ''))
                    self._cpp_id_map[cpp_id] = mem['id']

        # Populate HNSW index if present (H4: try persist-cache first, bulk-rebuild on miss)
        if self._hnsw is not None and all_mems:
            if not self._try_load_hnsw(expected_count=len(all_mems)):
                self._bulk_build_hnsw(all_mems)

    def _load_indexes_only(self):
        """For lazy_graph mode: populate C++/HNSW indexes without filling _graph_nodes."""
        all_mems = self.store.get_all()
        if self._cpp:
            for mem in all_mems:
                emb = mem.get('embedding', [])
                if emb and len(emb) == self.dim:
                    cpp_id = self._cpp.store(emb, mem.get('label', ''), mem.get('content', ''))
                    self._cpp_id_map[cpp_id] = mem['id']
        if self._hnsw is not None and all_mems:
            if not self._try_load_hnsw(expected_count=len(all_mems)):
                self._bulk_build_hnsw(all_mems)

    # -- H4: HNSW persistence helpers --------------------------------------

    def _try_load_hnsw(self, expected_count: int) -> bool:
        """Attempt to load HNSW index from disk. Returns True on success.

        Staleness check: loaded count must equal expected_count (from DB).
        Mismatches trigger a rebuild.
        """
        if self._hnsw is None or not self._hnsw_index_path:
            return False
        p = Path(self._hnsw_index_path)
        if not p.exists():
            return False
        try:
            # Init max_elements with generous headroom before loading
            self._hnsw_capacity = max(expected_count * 3 // 2, 1024)
            self._hnsw.load_index(str(p), max_elements=self._hnsw_capacity)
            loaded = self._hnsw.get_current_count()
            if loaded != expected_count:
                import logging
                logging.getLogger(__name__).debug(
                    "hnsw index stale: loaded=%d expected=%d; rebuilding",
                    loaded, expected_count,
                )
                # Re-init from scratch so bulk_build can add cleanly
                import hnswlib  # type: ignore
                self._hnsw = hnswlib.Index(space='cosine', dim=self.dim)
                self._hnsw.init_index(
                    max_elements=self._hnsw_capacity,
                    ef_construction=200, M=16,
                )
                self._hnsw.set_ef(self._hnsw_ef)
                return False
            self._hnsw_count = loaded
            return True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("hnsw load failed: %s", exc)
            return False

    def _bulk_build_hnsw(self, all_mems: list[dict]) -> None:
        """Build HNSW index from scratch via bulk add_items."""
        if self._hnsw is None or not all_mems:
            return
        try:
            self._hnsw_ensure_capacity(len(all_mems))
            import numpy as np
            vecs = np.asarray([m['embedding'] for m in all_mems], dtype=np.float32)
            ids = np.asarray([m['id'] for m in all_mems], dtype=np.int64)
            self._hnsw.add_items(vecs, ids)
            self._hnsw_count = len(all_mems)
            # Save so next startup is fast
            self._save_hnsw()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("hnsw bulk add failed: %s", exc)
            self._hnsw = None

    def _save_hnsw(self) -> None:
        """Persist HNSW index to disk. Called on close() + periodically in remember()."""
        if self._hnsw is None or not self._hnsw_index_path:
            return
        try:
            p = Path(self._hnsw_index_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._hnsw.save_index(str(p))
            self._hnsw_writes_since_save = 0
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("hnsw save failed: %s", exc)

    def _maybe_save_hnsw(self) -> None:
        """Save HNSW index if enough writes have accumulated since last save."""
        self._hnsw_writes_since_save += 1
        if self._hnsw_writes_since_save >= self._hnsw_save_every:
            self._save_hnsw()

    def _hnsw_ensure_capacity(self, incoming: int) -> None:
        """Grow HNSW capacity if the next insert would exceed it."""
        if self._hnsw is None:
            return
        needed = self._hnsw_count + incoming
        if needed > self._hnsw_capacity:
            # Grow with 1.5x headroom to amortize resize cost
            new_cap = max(needed * 3 // 2, self._hnsw_capacity * 2)
            try:
                self._hnsw.resize_index(new_cap)
                self._hnsw_capacity = new_cap
            except Exception:
                pass

    def _ensure_node(self, node_id: int) -> dict:
        """Lazy-load a single graph node from the store. Cached after first fetch."""
        if node_id in self._graph_nodes:
            return self._graph_nodes[node_id]
        mem = self.store.get(node_id)
        if not mem:
            return {}
        node = {'embedding': mem['embedding'], 'label': mem['label'], 'connections': {}}
        conns = self.store.get_connections(node_id)
        for c in conns:
            other = c['target'] if c['source'] == node_id else c['source']
            node['connections'][other] = c['weight']
        self._graph_nodes[node_id] = node
        return node
    
    # ------------------------------------------------------------------
    # Phase 7 Commit 3: entity registry public API
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        """Return entity dict (id, label, frequency, last_seen) or None."""
        return self.entities.get_entity(name) if self.entities else None

    def get_entities_for_memory(self, memory_id: int) -> list[dict[str, Any]]:
        """Return entity dicts linked to memory_id via mentions_entity edges."""
        return self.entities.get_entities_for_memory(memory_id) if self.entities else []

    def count_entities_named(self, name: str) -> int:
        """Case-insensitive count of entities with given name."""
        return self.entities.count_entities_named(name) if self.entities else 0

    # ------------------------------------------------------------------
    # Phase 7 Commit 5: sparse + temporal retrieval channels
    # ------------------------------------------------------------------

    def sparse_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """SQLite FTS5 BM25 retrieval. Returns memories whose content matches
        the FTS query, ranked by built-in BM25 relevance.

        FTS5's default whitespace-split treats multi-word queries as
        implicit-AND, requiring ALL terms in the same chunk. For natural-
        language queries that's too strict (caught post-ingest 2026-05-01:
        240 AE-domain queries returned 0 hits each). We tokenize on
        whitespace, strip punctuation, drop stopwords, and OR-join the
        terms so BM25 ranks chunks containing MORE of the query terms
        higher without requiring the conjunction.

        Returns empty list if FTS5 is unavailable on this SQLite build.
        """
        if not query or not query.strip():
            return []

        # Tokenize: keep alphanumerics + a few special chars common in AE
        # corpus (slash for "12/2 romex", hyphen for "lot-12"). Drop
        # punctuation, lowercase. Skip very short tokens + common stopwords.
        import re
        _SPARSE_STOPWORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "of", "in", "on", "at", "to", "for", "by",
            "with", "from", "as", "this", "that", "these", "those", "it",
            "its", "what", "when", "where", "why", "how", "who", "which",
            "do", "does", "did", "have", "has", "had", "will", "would",
            "should", "could", "can", "may", "might", "i", "we", "you",
            "they", "he", "she", "find", "show", "list", "give", "tell",
            "memories", "mention", "memory", "notes", "note", "about",
        })
        tokens = re.findall(r"[A-Za-z0-9/\-]+", query.lower())
        # Keep tokens with at least 2 alphanumerics, drop stopwords
        kept = [t for t in tokens
                if len(re.sub(r"[^A-Za-z0-9]", "", t)) >= 2
                and t not in _SPARSE_STOPWORDS]
        if not kept:
            return []
        # Quote each token (FTS5 phrase syntax) and OR-join
        # Use phrase quoting so tokens with hyphens/slashes match literally
        fts_query = " OR ".join(f'"{t}"' for t in kept)

        try:
            with self.store._lock:
                rows = self.store.conn.execute(
                    "SELECT rowid FROM memories_fts WHERE content MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, k),
                ).fetchall()
        except sqlite3.OperationalError:
            # If preprocessing produced an unparseable FTS5 query, fall back
            # to bare first-token search rather than returning empty.
            try:
                with self.store._lock:
                    rows = self.store.conn.execute(
                        "SELECT rowid FROM memories_fts WHERE content MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (kept[0], k),
                    ).fetchall()
            except sqlite3.OperationalError:
                return []
        results = []
        for row in rows:
            mem = self.store.get(row[0])
            if mem:
                results.append(mem)
        return results

    def temporal_search(self, query: str, as_of: float,
                        k: int = 5) -> list[dict[str, Any]]:
        """Semantic recall + bi-temporal validity filter at point-in-time `as_of`.

        Returns only memories whose [valid_from, valid_to] window contains
        `as_of`. NULL valid_from is treated as -infinity; NULL valid_to as
        +infinity (matches Graphiti's "always-valid" semantics for non-
        bi-temporal memories).

        Per addendum lines 318-330 + handoff section 7.5.
        """
        # Over-fetch so the temporal filter has candidates to work with.
        raw = self._recall_inner(query, max(k * 5, 25))
        if not raw:
            return []
        ids = [r['id'] for r in raw]
        placeholders = ",".join("?" * len(ids))
        with self.store._lock:
            rows = self.store.conn.execute(
                f"SELECT id, valid_from, valid_to FROM memories "
                f"WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        validity = {r[0]: (r[1], r[2]) for r in rows}
        valid_results = [
            r for r in raw
            if self._is_valid_at(*validity.get(r['id'], (None, None)), as_of=as_of)
        ]
        return valid_results[:k]

    # ------------------------------------------------------------------
    # Phase 7 Commit 7: PPR + MAGMA-style relation views
    # ------------------------------------------------------------------

    # Per handoff section 17.5. Intent → edge-type weight multipliers.
    # Single graph + filter weights; not separate physical graphs.
    _EDGE_WEIGHTS_BY_INTENT: dict[str, dict[str, float]] = {
        "factual":     {"semantic_similar_to": 0.8, "mentions_entity": 0.9,
                        "supports": 0.7, "contradicts": 0.3,
                        "happened_before": 0.4, "caused_by": 0.4},
        "causal":      {"semantic_similar_to": 0.5, "mentions_entity": 0.8,
                        "supports": 0.7, "contradicts": 0.6,
                        "happened_before": 0.8, "caused_by": 1.0},
        "temporal":    {"semantic_similar_to": 0.4, "mentions_entity": 0.7,
                        "happened_before": 1.0, "caused_by": 0.5},
        "procedural":  {"semantic_similar_to": 0.6, "mentions_entity": 0.5,
                        "summarizes": 0.4, "derived_from": 0.5,
                        "applies_to": 0.9},
        "entity":      {"semantic_similar_to": 0.5, "mentions_entity": 1.0,
                        "applies_to": 0.7, "located_in": 0.5},
    }

    @staticmethod
    def _classify_intent(query: str) -> str:
        """Heuristic intent classifier from query text. Returns one of the
        keys in _EDGE_WEIGHTS_BY_INTENT. Falls back to 'factual'."""
        if not query:
            return "factual"
        q = query.lower().strip()
        if q.startswith("who ") or " who " in q or "contact" in q:
            return "entity"
        if q.startswith("when ") or " when " in q or "before" in q or "after" in q:
            return "temporal"
        if q.startswith("why ") or " why " in q or "because" in q or "caused" in q:
            return "causal"
        if q.startswith("how ") or "how do" in q or "how to" in q or "how should" in q:
            return "procedural"
        return "factual"

    def intent_edge_weights(self, query: str) -> dict[str, float]:
        """Return edge-type weight multipliers for the query's classified intent.
        Used by graph_search to weight PPR traversal."""
        intent = self._classify_intent(query)
        return dict(self._EDGE_WEIGHTS_BY_INTENT[intent])

    def available_relation_views(self) -> list[str]:
        """Per addendum acceptance test: list of supported relation views."""
        return ["semantic", "temporal", "causal", "entity", "procedural"]

    @staticmethod
    def uses_single_connection_table() -> bool:
        """Per addendum acceptance test: confirms unified-graph substrate.
        We do NOT have separate semantic/temporal/causal/entity tables."""
        return True

    def graph_search(self, query: str, k: int = 5,
                     hops: int = 2) -> list[dict[str, Any]]:
        """PPR-style graph retrieval with intent-aware edge weighting.

        Strategy:
            1. Seed: dense recall finds top-k semantically similar memories
            2. Traverse: BFS up to `hops` from each seed, weighting edges by
               the query's classified intent (entity/temporal/causal/etc.)
            3. Score: accumulated activation = seed similarity × edge weight
               product per path; aggregated when multiple paths reach same node
            4. Return top-k by activation, deduplicated against seeds

        Single connections table; relation views are weight filters, not
        separate stores (per non-negotiable handoff constraint 2.1).
        """
        weights = self.intent_edge_weights(query)
        seeds = self._recall_inner(query, k, temporal_weight=0.2)
        if not seeds:
            return []

        # node_id -> max activation seen. Per Reviewer #1: defensively use
        # .get() so malformed seed/edge dicts don't KeyError the whole
        # traversal (e.g., partially-constructed entries from a future
        # cross-encoder rerank step).
        activation: dict[int, float] = {}
        for seed in seeds:
            sid = seed.get('id')
            if sid is None:
                continue
            base = float(seed.get('similarity', seed.get('combined', 0.5)))
            activation[sid] = max(activation.get(sid, 0.0), base)

            # BFS up to `hops` levels
            frontier = {sid: base}
            for _ in range(hops):
                next_frontier: dict[int, float] = {}
                for node_id, node_act in frontier.items():
                    try:
                        edges = self.store.get_connections(node_id)
                    except Exception:
                        continue
                    for edge in edges:
                        src = edge.get('source')
                        tgt = edge.get('target')
                        if src is None or tgt is None:
                            continue
                        other = tgt if src == node_id else src
                        # get_connections() returns dict key 'type' (NOT 'edge_type')
                        et = edge.get('type') or edge.get('edge_type') or 'semantic_similar_to'
                        w = weights.get(et, 0.3)  # unknown edge-type baseline weight
                        new_act = node_act * w * 0.7  # 0.7 = damping factor
                        if new_act > next_frontier.get(other, 0.0):
                            next_frontier[other] = new_act
                        if new_act > activation.get(other, 0.0):
                            activation[other] = new_act
                frontier = next_frontier

        # Rank by accumulated activation. Seeds remain in results — nodes
        # that are BOTH semantically similar AND graph-reachable get the
        # highest activation, which is the desired behavior. Caller can
        # post-filter against recall() output if they want pure graph hits.
        ranked = sorted(
            activation.items(),
            key=lambda x: -x[1],
        )
        results = []
        for node_id, act in ranked[:k]:
            mem = self.store.get(node_id)
            if mem:
                mem['activation'] = round(act, 4)
                results.append(mem)
        return results

    @staticmethod
    def _is_valid_at(valid_from: Optional[float], valid_to: Optional[float],
                     as_of: float) -> bool:
        """Bi-temporal validity check. NULL bounds are treated as unbounded
        (always-valid semantics for memories without explicit validity)."""
        if valid_from is not None and valid_from > as_of:
            return False
        if valid_to is not None and as_of >= valid_to:
            return False
        return True

    @staticmethod
    def _normalize_dates(text: str, ref_time: float = None) -> str:
        """H18: replace relative dates ("yesterday", "last week", "N days ago")
        with absolute ISO dates. Conservative — leaves ambiguous phrasings
        ("a few days ago", "shortly") untouched.

        Parallels Anthropic Auto-Dream's Consolidate-phase date normalization:
        "Yesterday we decided to use Redis" → "On 2026-04-24 we decided to use Redis".
        """
        if not text:
            return text
        import re
        from datetime import datetime, timedelta
        ref = datetime.fromtimestamp(ref_time) if ref_time else datetime.now()

        def _iso(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%d")

        def _sub_static(match, days_offset):
            return f"on {_iso(ref + timedelta(days=days_offset))}"

        def _sub_n(match, factor):
            n = int(match.group(1))
            return f"on {_iso(ref - timedelta(days=n * factor))}"

        # Static patterns (case-insensitive)
        static = [
            (r"\byesterday\b", -1),
            (r"\btoday\b", 0),
            (r"\btomorrow\b", 1),
            (r"\bearlier today\b", 0),
            (r"\bthis morning\b", 0),
            (r"\bthis afternoon\b", 0),
            (r"\bthis evening\b", 0),
            (r"\btonight\b", 0),
            (r"\blast week\b", -7),
            (r"\bnext week\b", 7),
            (r"\blast month\b", -30),
            (r"\bnext month\b", 30),
        ]
        out = text
        for pat, offset in static:
            out = re.sub(pat, lambda m, o=offset: _sub_static(m, o), out, flags=re.IGNORECASE)

        # N-day / N-week ago patterns
        out = re.sub(r"\b(\d+)\s+days?\s+ago\b",
                     lambda m: _sub_n(m, 1), out, flags=re.IGNORECASE)
        out = re.sub(r"\b(\d+)\s+weeks?\s+ago\b",
                     lambda m: _sub_n(m, 7), out, flags=re.IGNORECASE)

        return out

    def remember(self, text: str, label: str = "", detect_conflicts: bool = True,
                 normalize_dates: bool = True,
                 *,
                 kind: Optional[str] = None,
                 confidence: Optional[float] = None,
                 source: Optional[str] = None,
                 origin_system: Optional[str] = None,
                 valid_from: Optional[float] = None,
                 valid_to: Optional[float] = None,
                 metadata: Optional[dict[str, Any]] = None,
                 evidence_ids: Optional[list[int]] = None,
                 salience: Optional[float] = None,
                 procedural_score: Optional[float] = None) -> int:
        """Store a memory. Returns memory ID.

        If detect_conflicts=True, checks for existing memories about the same
        topic that contain contradictory information and updates/invalidates them.

        If normalize_dates=True (default; H18), relative dates in the content
        are converted to absolute ISO dates before storage. Pass False to
        preserve verbatim text.

        Phase 7 typed-memory kwargs (all optional, all keyword-only):
        - kind: one of memory_types.MEMORY_KINDS. Auto-classified from text
          via classify_memory_kind() when not provided.
        - confidence: caller-supplied confidence in [0,1]; defaults to schema
          default (1.0).
        - source: provenance label (e.g., 'whatsapp', 'estimate_form', 'cli').
        - origin_system: which system wrote this memory (e.g., 'ae', 'hermes').
        - valid_from / valid_to: bi-temporal validity window (Graphiti-style).
        - metadata: dict, JSON-serialized into metadata_json column.

        Note: H19 supersession path (when detect_conflicts fires) does NOT yet
        propagate typed kwargs through replace_memory(). Superseded memories
        retain pre-supersession typing. Tracked for Commit 3.
        """
        import math
        import time

        if normalize_dates:
            text = self._normalize_dates(text)

        # Auto-classify kind from text when caller did not specify.
        if kind is None:
            kind = classify_memory_kind(text, metadata=metadata)

        # Phase 7.5-α: auto-populate procedural_score so the unified scorer's
        # procedural channel actually contributes to ranking. Pre-fix the
        # column was always NULL → 0.0 in scorer features, which made
        # `+ w_procedural * f.procedural_score` always 0. Baseline 0.7 for
        # kind='procedural'; None (NULL) for everything else. Future
        # refinement: compute from text features (imperative-verb ratio,
        # code-block density, list-shape, etc.). Backfill for existing
        # procedural rows is one-shot via tools/backfill_procedural_score.py.
        if procedural_score is None and kind == "procedural":
            procedural_score = 0.7

        embedding = self.embedder.embed(text)
        
        # Knowledge-update: detect conflicts with existing memories
        if detect_conflicts and self._graph_nodes:
            conflicts = self._find_conflicts(text, embedding)
            for conflict_id, conflict_info in conflicts.items():
                old_content = conflict_info['content']
                similarity = conflict_info['similarity']
                
                # High similarity + different content = likely update
                if similarity > 0.7 and self._content_differs(old_content, text):
                    now_ts = time.time()
                    # H19: archive old content to superseded_memories, then
                    # cleanly replace the memories row. No more [SUPERSEDED]
                    # prefix. Audit history preserved separately.
                    try:
                        # Fetch the full old row for archive (need embedding + label + salience)
                        old_full = self.store.get(conflict_id) or {}
                        old_embedding = old_full.get('embedding', [])
                        old_label = old_full.get('label', '') or conflict_info.get('label', '')
                        old_salience = old_full.get('salience', 1.0)
                        self.store.archive_superseded(
                            original_id=conflict_id,
                            content=old_content,
                            label=old_label,
                            embedding=old_embedding,
                            salience=old_salience,
                            superseded_by=conflict_id,  # same id; new content will be there
                            superseded_at=now_ts,
                            superseded_reason=f"cosine={similarity:.3f} content_differs",
                        )
                        # Phase 7 C11 carryover: propagate typed kwargs from
                        # the user's remember() call into the supersession
                        # replacement. transaction_time auto-stamps to now if
                        # not provided.
                        self.store.replace_memory(
                            memory_id=conflict_id,
                            content=text,
                            label=label or text[:60],
                            embedding=embedding,
                            kind=kind,
                            confidence=confidence,
                            source=source,
                            origin_system=origin_system,
                            valid_from=valid_from,
                            valid_to=valid_to,
                            transaction_time=time.time(),
                            metadata=metadata,
                        )
                    except Exception as exc:
                        # Fall back to legacy [SUPERSEDED] prefix if anything
                        # unexpected happens (defensive — keeps recall semantics)
                        import logging
                        logging.getLogger(__name__).warning(
                            "H19 supersede archive failed, falling back to prefix: %s", exc,
                        )
                        with self.store._lock:
                            self.store.conn.execute(
                                "UPDATE memories SET content = ? WHERE id = ?",
                                (f"[SUPERSEDED] {old_content}\n[UPDATED TO] {text}", conflict_id),
                            )
                            self.store.conn.commit()
                    # H3: invalidate old edges temporally (without deleting).
                    try:
                        self.store.set_edges_valid_to(conflict_id, now_ts)
                    except Exception:
                        pass
                    # Update in-memory graph
                    if conflict_id in self._graph_nodes:
                        self._graph_nodes[conflict_id]['embedding'] = embedding
                        # Clear stale in-memory edges since DB marked them expired
                        self._graph_nodes[conflict_id]['connections'] = {}
                        # Remove references from other nodes' connections
                        for other_id, other_node in self._graph_nodes.items():
                            other_node.get('connections', {}).pop(conflict_id, None)
                    # Don't create duplicate - update existing
                    return conflict_id
        
        mem_id = self.store.store(
            label or text[:60], text, embedding,
            kind=kind,
            confidence=confidence,
            source=source,
            origin_system=origin_system,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            salience=salience,
            procedural_score=procedural_score,
        )

        # Phase 7 Commit 3: extract entities from text + create mentions_entity edges.
        # Skip for entity-kind writes (avoid recursive entity-of-entity loops).
        if self.entities is not None and kind != "entity":
            try:
                self.entities.process_memory(mem_id, text)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "entity extraction failed for memory %d: %s", mem_id, exc,
                )

        # Phase 7 Commit 4: derived_from edges from new memory back to supporting
        # experiences. Used by procedural memory to cite its evidence base, and
        # by dream-insight nodes to point at source memories. Invalid evidence
        # IDs are silently skipped (best-effort link).
        if evidence_ids:
            for ev_id in evidence_ids:
                try:
                    self.store.add_connection(
                        source=mem_id,
                        target=int(ev_id),
                        weight=1.0,
                        edge_type="derived_from",
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "evidence link failed (mem=%d ev=%s): %s", mem_id, ev_id, exc,
                    )
        
        # Add to in-memory graph
        self._graph_nodes[mem_id] = {
            'embedding': embedding,
            'label': label or text[:60],
            'connections': {}
        }

        # Add to C++ SIMD index + track mapping
        if self._cpp:
            try:
                cpp_id = self._cpp.store(embedding, label or text[:60], text)
                self._cpp_id_map[cpp_id] = mem_id
            except Exception:
                pass

        # Add to HNSW index (if active)
        if self._hnsw is not None:
            try:
                import numpy as np
                self._hnsw_ensure_capacity(1)
                self._hnsw.add_items(
                    np.asarray([embedding], dtype=np.float32),
                    np.asarray([mem_id], dtype=np.int64),
                )
                self._hnsw_count += 1
                # H4: periodic persistence so we're not rebuilding from scratch on every startup
                self._maybe_save_hnsw()
            except Exception:
                pass

        # Auto-connect to similar memories
        for other_id, other_node in self._graph_nodes.items():
            if other_id == mem_id:
                continue
            sim = self._cosine_similarity(embedding, other_node['embedding'])
            if sim > 0.15:  # Threshold for connection
                self._graph_nodes[mem_id]['connections'][other_id] = sim
                self._graph_nodes[other_id]['connections'][mem_id] = sim
                self.store.add_connection(mem_id, other_id, sim)

        return mem_id
    
    def _find_conflicts(self, new_text: str, new_embedding: list[float], threshold: float = 0.6) -> dict:
        """Find memories that might conflict with the new text.
        Returns {memory_id: {similarity, content}} for potential conflicts.
        """
        conflicts = {}
        for mem in self.store.get_all():
            sim = self._cosine_similarity(new_embedding, mem['embedding'])
            if sim > threshold:
                conflicts[mem['id']] = {
                    'similarity': sim,
                    'content': mem['content'],
                    'label': mem['label']
                }
        return conflicts
    
    def _content_differs(self, old_text: str, new_text: str) -> bool:
        """Check if two texts contain different information despite high similarity.
        Heuristics: different numbers, dates, negations, or significant word differences.
        """
        import re
        
        old_clean = old_text.replace("[SUPERSEDED]", "").replace("[UPDATED TO]", "").strip()
        
        # Extract numbers from both
        old_nums = set(re.findall(r'\d+\.?\d*', old_clean))
        new_nums = set(re.findall(r'\d+\.?\d*', new_text))
        
        # Different numbers = different info
        if old_nums != new_nums and old_nums and new_nums:
            return True
        
        # Check for negation differences
        negations = {'not', "n't", 'never', 'no', 'none', 'nothing', 'nowhere'}
        old_neg = any(n in old_clean.lower().split() for n in negations)
        new_neg = any(n in new_text.lower().split() for n in negations)
        if old_neg != new_neg:
            return True
        
        # Check for date differences
        old_dates = set(re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', old_clean))
        new_dates = set(re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', new_text))
        if old_dates != new_dates and old_dates and new_dates:
            return True
        
        # Check for significant content word differences (excluding common words)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'it', 'its', "it's", 'this', 'that',
                     'user', 'user\'s', 'my', 'i', 'me', 'we', 'our', 'you', 'your'}
        
        def extract_keywords(text):
            words = set(re.findall(r'\b[a-z]+\b', text.lower()))
            return words - stopwords
        
        old_kw = extract_keywords(old_clean)
        new_kw = extract_keywords(new_text)
        
        # If more than 30% of keywords differ, it's a real update
        if old_kw and new_kw:
            shared = old_kw & new_kw
            total = old_kw | new_kw
            diff_ratio = 1.0 - len(shared) / len(total)
            if diff_ratio > 0.3:
                return True
        
        return False
    
    def _maybe_rerank(self, query: str, candidates: list[dict],
                      force: bool = False) -> list[dict]:
        """Cross-encoder rerank of a candidate set. No-op if disabled or model unavailable.

        Candidates must carry 'content' (string). Preserves all other fields and
        adds 'rerank_score'. Re-sorts by rerank_score descending.

        Caught 2026-05-01: previously gated only on self._rerank, so
        callers passing rerank=True per-call (like the AE-domain bench
        harness) had their kwarg silently ignored. force=True overrides
        the instance flag and lets per-call rerank requests through.
        """
        if not candidates:
            return candidates
        if not (force or self._rerank):
            return candidates
        try:
            if self._rerank_model is None:
                from sentence_transformers import CrossEncoder
                self._rerank_model = CrossEncoder(self._rerank_model_name)
            pairs = [(query, c.get('content') or c.get('label') or '') for c in candidates]
            scores = self._rerank_model.predict(pairs)
            for c, s in zip(candidates, scores):
                c['rerank_score'] = float(s)
            candidates.sort(key=lambda x: -x.get('rerank_score', 0.0))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("rerank skipped: %s", exc)
        return candidates

    @staticmethod
    def _effective_salience(salience_base, access_count, created_at, now=None) -> float:
        """Compute a runtime salience factor from stored base + access + age.

        Kept non-persistent so we don't serialize write contention into recall.
        - Age decay: gentle exp(-k*days)
        - Access boost: log1p(access_count) * alpha
        - Clamped to [SALIENCE_MIN, SALIENCE_MAX]
        """
        import math
        import time as _time
        if now is None:
            now = _time.time()
        base = salience_base if salience_base is not None else 1.0
        age_days = max(0.0, (now - (created_at or now)) / 86400.0)
        decay = math.exp(-SALIENCE_AGE_DECAY_K * age_days)
        boost = math.log1p(max(0, access_count or 0)) * SALIENCE_ACCESS_BOOST
        eff = base * decay + boost
        return max(SALIENCE_MIN, min(SALIENCE_MAX, eff))

    def recall(self, query: str, k: int = 5, temporal_weight: float = 0.2,
               *, kind: Optional[str] = None,
               as_of: Optional[float] = None) -> list[dict]:
        """Retrieve memories related to query.

        Args:
            query: Search query
            k: Number of results
            temporal_weight: Weight for recency scoring
            kind: Optional Phase 7 kind filter
            as_of: Optional bi-temporal point-in-time. When set, results are
                   filtered to memories valid at as_of (delegates to the
                   temporal channel + applies kind filter if both are set).

        Returns list of {id, label, content, similarity, temporal_score, ...}.
        """
        # Plain path: no Phase 7 filters — preserve pre-Phase-7 behavior.
        if kind is None and as_of is None:
            return self._recall_inner(query, k, temporal_weight)

        # Over-fetch to compensate for filtering loss.
        raw = self._recall_inner(query, max(k * 5, 25), temporal_weight)
        if not raw:
            return []

        # Apply kind filter if requested.
        if kind is not None:
            raw = self._filter_by_kind(raw, kind)

        # Apply as_of validity filter if requested.
        if as_of is not None and raw:
            ids = [r['id'] for r in raw]
            placeholders = ",".join("?" * len(ids))
            with self.store._lock:
                rows = self.store.conn.execute(
                    f"SELECT id, valid_from, valid_to FROM memories "
                    f"WHERE id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
            validity = {r[0]: (r[1], r[2]) for r in rows}
            raw = [
                r for r in raw
                if self._is_valid_at(*validity.get(r['id'], (None, None)), as_of=as_of)
            ]

        return raw[:k]

    def scoring_config(self):
        """Return the unified scoring config — final_authority and feature
        list. Used to verify (per addendum acceptance test) that RRF is a
        feature, never the final ranking law.
        """
        from scoring import ScoringConfig
        return ScoringConfig()

    # ------------------------------------------------------------------
    # Phase 7 Commit 11: hybrid retrieval — Hindsight-shape candidate
    # union + salience-weighted continuous final law
    # ------------------------------------------------------------------

    def hybrid_recall(self, query: str, k: int = 5,
                     *,
                     as_of: Optional[float] = None,
                     kind: Optional[str] = None,
                     hops: int = 2,
                     pool_per_channel: int = 25,
                     rerank: Optional[bool] = None) -> list[dict[str, Any]]:
        """Multi-channel hybrid retrieval. Borrowed Hindsight's pool-union
        shape (dense + sparse + graph + temporal candidates) but applied
        the salience-weighted continuous scoring law instead of pure RRF.

        Strategy:
            1. Pool: union candidate IDs from up to 4 channels (each
               returns up to `pool_per_channel` items).
            2. Per-channel ranks → RRF feature (one of many features,
               NOT the final authority).
            3. Build CandidateFeatures with all 12 fields populated (per
               Phase 7.5 α/β/γ/δ/ε wiring):
                 - Channel scores: semantic, sparse, graph, temporal
                 - Entity boost (β): entity_score from mentions_entity edges
                 - Procedural boost (α): procedural_score from memories col
                 - Locus boost (ε): locus_score from located_in edges
                 - Memory bias: salience, confidence
                 - Rank-based: rrf_feature
                 - Penalties (γ, δ): stale_penalty (last_accessed age),
                   contradiction_penalty (contradicts edge count × 0.05)
            4. Final score via scoring.score_candidate() with intent-aware
               edge weights for the graph channel.
            5. Optional cross-encoder rerank on top-N (uses self._rerank
               flag if set; can override via rerank kwarg).
            6. Apply kind/as_of post-filters if requested.
            7. Return top k. Each result carries:
                 - 'combined' (final score)
                 - 'channels' (which channels surfaced this candidate)
                 - '_trace' (per-feature contribution dict;
                   mazemaker-inspired explainability — Phase 7.5)

        Path to Hindsight 0.92+ R@5 from current 0.53 baseline:
            - Install FlagEmbedding for BGE-M3 hybrid embedding (~1GB)
            - Enable cross-encoder rerank (sentence-transformers required)
            - Label AE-domain ground truth
            - Tune DEFAULT_WEIGHTS against labeled bench
        Without those, this method ships the architectural unification —
        same retrieval shape as Hindsight, different final-scoring law.
        """
        from scoring import CandidateFeatures, DEFAULT_WEIGHTS, score_candidate

        # ---- Pool union from up to 4 channels --------------------------
        per_channel_ranks: dict[str, dict[int, int]] = {}
        per_channel_scores: dict[str, dict[int, float]] = {}

        # Channel 1: dense semantic (use over-fetched recall for the pool)
        dense_results = self._recall_inner(query, pool_per_channel)
        per_channel_ranks["semantic"] = {
            r["id"]: idx for idx, r in enumerate(dense_results)
        }
        per_channel_scores["semantic"] = {
            r["id"]: float(r.get("similarity", 0.0)) for r in dense_results
        }

        # Channel 2: sparse FTS5
        sparse_results = self.sparse_search(query, k=pool_per_channel)
        per_channel_ranks["sparse"] = {
            r["id"]: idx for idx, r in enumerate(sparse_results)
        }
        # FTS5 doesn't return numeric similarity in our wrapper; use
        # rank-derived score (1.0 at top, decays linearly)
        per_channel_scores["sparse"] = {
            r["id"]: 1.0 - (idx / max(len(sparse_results), 1))
            for idx, r in enumerate(sparse_results)
        }

        # Channel 3: graph PPR with intent-aware weights
        graph_results = self.graph_search(query, k=pool_per_channel, hops=hops)
        per_channel_ranks["graph"] = {
            r["id"]: idx for idx, r in enumerate(graph_results)
        }
        per_channel_scores["graph"] = {
            r["id"]: float(r.get("activation", 0.0)) for r in graph_results
        }

        # Channel 4: temporal validity (only if as_of given)
        temporal_ids: set[int] = set()
        if as_of is not None:
            temporal_results = self.temporal_search(query, as_of=as_of,
                                                    k=pool_per_channel)
            per_channel_ranks["temporal"] = {
                r["id"]: idx for idx, r in enumerate(temporal_results)
            }
            per_channel_scores["temporal"] = {
                r["id"]: float(r.get("similarity", 0.0))
                for r in temporal_results
            }
            temporal_ids = set(r["id"] for r in temporal_results)

        # ---- Candidate union -------------------------------------------
        candidate_ids = (
            set(per_channel_ranks.get("semantic", {}).keys())
            | set(per_channel_ranks.get("sparse", {}).keys())
            | set(per_channel_ranks.get("graph", {}).keys())
            | temporal_ids
        )
        if not candidate_ids:
            return []

        # ---- Fetch typed-row metadata in one batched query -------------
        placeholders = ",".join("?" * len(candidate_ids))
        with self.store._lock:
            rows = self.store.conn.execute(
                f"SELECT id, salience, confidence, kind, valid_from, valid_to, "
                f"       procedural_score, last_reinforced_at, last_accessed, "
                f"       created_at "
                f"FROM memories WHERE id IN ({placeholders})",
                tuple(sorted(candidate_ids)),
            ).fetchall()
        meta = {r[0]: {"salience": r[1] or 1.0, "confidence": r[2] or 1.0,
                       "kind": r[3], "valid_from": r[4], "valid_to": r[5],
                       "procedural_score": r[6] or 0.0,
                       "last_reinforced_at": r[7],
                       "last_accessed": r[8],
                       "created_at": r[9]}
                for r in rows}

        # ---- Phase 7.5-β: per-candidate entity_score ------------------
        # If query mentions entities and a candidate's mentions_entity edges
        # overlap them, boost via the entity channel. Caught 2026-05-01:
        # CandidateFeatures.entity_score had been a no-op because the call
        # site never populated it. This batches all candidate-edge lookups
        # into one IN-query for performance.
        entity_score_by_id: dict[int, float] = {}
        try:
            from entity_extraction import extract_entities
            query_entities = {e.lower() for e in extract_entities(query)}
            if query_entities:
                with self.store._lock:
                    edge_rows = self.store.conn.execute(
                        f"SELECT c.source_id, m.label "
                        f"FROM connections c JOIN memories m ON m.id = c.target_id "
                        f"WHERE c.edge_type = 'mentions_entity' "
                        f"  AND m.kind = 'entity' "
                        f"  AND c.source_id IN ({placeholders})",
                        tuple(sorted(candidate_ids)),
                    ).fetchall()
                cand_entities: dict[int, set[str]] = {}
                for src_id, ent_label in edge_rows:
                    if ent_label:
                        cand_entities.setdefault(src_id, set()).add(
                            ent_label.lower()
                        )
                qmax = max(len(query_entities), 1)
                for cid, ents in cand_entities.items():
                    overlap = len(ents & query_entities)
                    if overlap > 0:
                        entity_score_by_id[cid] = min(overlap / qmax, 1.0)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "entity_score wiring failed silently: %s", exc,
            )

        # ---- Phase 7.5-ε: per-candidate locus_score ------------------
        # If query mentions a locus name and a candidate has a located_in
        # edge to that locus, boost via the locus channel. 0 locus rows
        # in live DB today (deferred AE-domain wiring), but ready to fire
        # once AE workflow seeds Lennar lots / customer addresses / job
        # sites as kind='locus' nodes. Same batched-query pattern as
        # entity_score.
        locus_score_by_id: dict[int, float] = {}
        try:
            # Single lock acquisition matching β/δ patterns (per round-5
            # reviewer cross-cutting findings).
            with self.store._lock:
                locus_rows = self.store.conn.execute(
                    "SELECT id, label FROM memories WHERE kind='locus'"
                ).fetchall()
                if locus_rows:
                    query_lower = query.lower()
                    matching_locus_ids = {
                        lid for lid, label in locus_rows
                        if label and label.lower() in query_lower
                    }
                    if matching_locus_ids:
                        locus_placeholders = ",".join(
                            "?" * len(matching_locus_ids)
                        )
                        loc_edge_rows = self.store.conn.execute(
                            f"SELECT source_id FROM connections "
                            f"WHERE edge_type='located_in' "
                            f"  AND source_id IN ({placeholders}) "
                            f"  AND target_id IN ({locus_placeholders})",
                            tuple(sorted(candidate_ids))
                            + tuple(sorted(matching_locus_ids)),
                        ).fetchall()
                        for (src_id,) in loc_edge_rows:
                            locus_score_by_id[src_id] = 1.0
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "locus_score wiring failed silently: %s", exc,
            )

        # ---- Phase 7.5-δ: per-candidate contradicts-edge count -------
        # Wired even though contradicts edges currently have 0 rows,
        # so the contradiction-detection runner ships from D5 onward,
        # this scorer instantly discounts memories with active conflicts.
        contradicts_count_by_id: dict[int, int] = {}
        try:
            with self.store._lock:
                cedge_rows = self.store.conn.execute(
                    f"SELECT source_id, COUNT(*) "
                    f"FROM connections "
                    f"WHERE edge_type='contradicts' "
                    f"  AND source_id IN ({placeholders}) "
                    f"GROUP BY source_id",
                    tuple(sorted(candidate_ids)),
                ).fetchall()
            for src_id, cnt in cedge_rows:
                contradicts_count_by_id[src_id] = cnt or 0
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "contradicts edge count failed silently: %s", exc,
            )

        # ---- Apply kind / as_of post-filters at candidate level --------
        if kind is not None:
            candidate_ids = {cid for cid in candidate_ids
                              if meta.get(cid, {}).get("kind") == kind}
        if as_of is not None:
            def _vp(cid: int) -> bool:
                m = meta.get(cid, {})
                return self._is_valid_at(m.get("valid_from"),
                                          m.get("valid_to"), as_of=as_of)
            candidate_ids = {cid for cid in candidate_ids if _vp(cid)}

        # ---- RRF feature per candidate (rank-based, normalized) --------
        # Bounded contribution; never the final authority.
        K_RRF = 60.0
        def _rrf(cid: int) -> float:
            score = 0.0
            for ch_ranks in per_channel_ranks.values():
                if cid in ch_ranks:
                    score += 1.0 / (K_RRF + ch_ranks[cid])
            # Normalize by max possible (1/K_RRF * num_channels)
            max_possible = (1.0 / K_RRF) * len(per_channel_ranks)
            return score / max_possible if max_possible > 0 else 0.0

        # ---- Continuous scoring (the final authority) ------------------
        scored: list[tuple[int, float, dict]] = []
        for cid in candidate_ids:
            m = meta.get(cid, {})
            # Phase 7.5-γ: stale_penalty for memories not reinforced/accessed
            # in a long time. Mild discount; capped at 0.3 to avoid
            # dominating the formula. Reference timestamp priority:
            #   1. last_reinforced_at (Phase 7 column; not yet written by
            #      production paths but ready for future reinforcement
            #      tracking)
            #   2. last_accessed (updated by SQLiteStore.get() on every
            #      retrieval — measures "haven't been queried recently")
            #   3. created_at (fresh-row fallback)
            # This means stale_penalty currently approximates "haven't
            # been read in N days" — semantically correct for stale
            # memories that aren't being touched anymore.
            ref_ts = (m.get("last_reinforced_at")
                      or m.get("last_accessed")
                      or m.get("created_at")
                      or 0.0)
            try:
                age_days = max((time.time() - float(ref_ts)) / 86400.0, 0.0)
            except (TypeError, ValueError):
                age_days = 0.0
            # Linear ramp from 0 (fresh) to 0.3 (90+ days untouched)
            stale_penalty = min(age_days / 300.0, 0.3) if age_days > 30 else 0.0

            # Phase 7.5-δ: contradiction_penalty from contradicts-edge count.
            # Currently 0 in live DB (the contradicts edge type has no
            # production consumer yet) — but wired so when a real
            # contradiction-detection runner ships, the scorer immediately
            # discounts memories with active contradictions.
            #
            # Magic 0.05: per-edge penalty. Calibrated to match the
            # procedural channel weight (DEFAULT_WEIGHTS["procedural"] = 0.05)
            # so a single contradiction roughly cancels a procedural boost.
            # Memories with 6+ contradicts edges hit -0.30 (matches stale_
            # penalty cap). Tunable; no test depends on the exact value.
            CONTRADICTION_PER_EDGE = 0.05
            contradiction_penalty = float(
                contradicts_count_by_id.get(cid, 0)
            ) * CONTRADICTION_PER_EDGE

            features = CandidateFeatures(
                memory_id=cid,
                semantic_score=per_channel_scores.get("semantic", {}).get(cid, 0.0),
                sparse_score=per_channel_scores.get("sparse", {}).get(cid, 0.0),
                graph_score=per_channel_scores.get("graph", {}).get(cid, 0.0),
                temporal_score=per_channel_scores.get("temporal", {}).get(cid, 0.0),
                procedural_score=float(m.get("procedural_score") or 0.0),
                entity_score=float(entity_score_by_id.get(cid, 0.0)),
                locus_score=float(locus_score_by_id.get(cid, 0.0)),
                stale_penalty=stale_penalty,
                contradiction_penalty=contradiction_penalty,
                rrf_feature=_rrf(cid),
                salience=float(m.get("salience", 1.0)),
                confidence=float(m.get("confidence", 1.0)),
            )
            final = score_candidate(features, DEFAULT_WEIGHTS)
            scored.append((cid, final, m, features))

        scored.sort(key=lambda x: -x[1])

        # ---- Optional cross-encoder rerank (Hindsight-style) -----------
        use_rerank = rerank if rerank is not None else getattr(self, "_rerank", False)
        if use_rerank and len(scored) > 0:
            top_n = min(50, len(scored))
            top_candidates = []
            for cid, _final, _m, _f in scored[:top_n]:
                row = self.store.get(cid)
                if row:
                    row["_combined"] = _final
                    top_candidates.append(row)
            if top_candidates:
                top_candidates = self._maybe_rerank(query, top_candidates,
                                                   force=use_rerank)
                # Splice reranked top back into scored in their new order
                reranked_ids = [c["id"] for c in top_candidates]
                tail_ids = [t[0] for t in scored[top_n:]]
                ordered_ids = reranked_ids + tail_ids
                id_to_meta = {t[0]: t for t in scored}
                scored = [id_to_meta[i] for i in ordered_ids if i in id_to_meta]

        # ---- Materialize top-k full memory rows ------------------------
        # Mazemaker-inspired (2026-05-01): activation trace as first-class
        # field. Each result now carries a `_trace` dict with the per-channel
        # contribution that produced its rank. Lets downstream UIs explain
        # "why did this rank?" without re-querying. See reference_mazemaker_
        # digest_2026-05-01.md for the pattern.
        results: list[dict[str, Any]] = []
        for cid, final_score, _m, features in scored[:k]:
            row = self.store.get(cid)
            if row:
                row["combined"] = round(final_score, 4)
                row["channels"] = [ch for ch in per_channel_ranks
                                    if cid in per_channel_ranks[ch]]
                row["_trace"] = {
                    "semantic":              round(features.semantic_score, 4),
                    "sparse":                round(features.sparse_score, 4),
                    "graph":                 round(features.graph_score, 4),
                    "temporal":              round(features.temporal_score, 4),
                    "entity":                round(features.entity_score, 4),
                    "procedural":            round(features.procedural_score, 4),
                    "locus":                 round(features.locus_score, 4),
                    "rrf_feature":           round(features.rrf_feature, 4),
                    "salience":              round(features.salience, 4),
                    "confidence":            round(features.confidence, 4),
                    "stale_penalty":         round(features.stale_penalty, 4),
                    "contradiction_penalty": round(features.contradiction_penalty, 4),
                }
                results.append(row)
        return results

    # ------------------------------------------------------------------
    # Phase 7 Commit 9: dream Memify + insight + contradiction hygiene
    # ------------------------------------------------------------------

    _CONTRADICTION_STOPWORDS = frozenset({
        "is", "are", "was", "were", "the", "a", "an", "and", "or", "but",
        "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
        "be", "been", "being", "do", "does", "did", "have", "has", "had",
        "this", "that", "these", "those", "it", "its",
    })

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        """Return memory row including salience + kind + content. Convenience
        wrapper around store.get() that adds the typed/scoring fields."""
        with self.store._lock:
            row = self.store.conn.execute(
                "SELECT id, label, content, salience, access_count, kind, "
                "confidence, valid_from, valid_to, transaction_time, "
                "origin_system, source, memory_visibility, pin_state, "
                "metadata_json FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "label": row[1], "content": row[2],
            "salience": row[3], "access_count": row[4], "kind": row[5],
            "confidence": row[6], "valid_from": row[7], "valid_to": row[8],
            "transaction_time": row[9], "origin_system": row[10],
            "source": row[11], "memory_visibility": row[12],
            "pin_state": row[13], "metadata_json": row[14],
        }

    def get_edges(self, memory_id: int) -> list[dict[str, Any]]:
        """Return edges touching memory_id. Maps get_connections's 'type'
        key to 'edge_type' for forward-compat with addendum tests."""
        edges = self.store.get_connections(memory_id, include_expired=True)
        out = []
        for e in edges:
            out.append({
                "source_id": e["source"],
                "target_id": e["target"],
                "weight": e["weight"],
                "edge_type": e.get("type") or "similar",
                "valid_from": e.get("valid_from"),
                "valid_to": e.get("valid_to"),
            })
        return out

    def has_edge(self, source_id: int, target_id: int,
                 edge_type: Optional[str] = None) -> bool:
        """True if an edge exists between source and target. If edge_type is
        given, additionally require matching type. Direction-insensitive."""
        sql = (
            "SELECT 1 FROM connections "
            "WHERE ((source_id = ? AND target_id = ?) "
            "    OR (source_id = ? AND target_id = ?))"
        )
        params: list[Any] = [source_id, target_id, target_id, source_id]
        if edge_type is not None:
            sql += " AND edge_type = ?"
            params.append(edge_type)
        sql += " LIMIT 1"
        with self.store._lock:
            row = self.store.conn.execute(sql, tuple(params)).fetchone()
        return row is not None

    def run_memify_once(self, decay_factor: float = 0.5) -> dict[str, int]:
        """Detect exact-content duplicates and downweight salience of all
        but the highest-salience copy. Per handoff sec 9.4 — dream Memify
        does not hard-delete; it downweights with provenance.

        Returns stats dict with downweighted count.
        """
        # Per Reviewer #1: race window between SELECT and UPDATE — concurrent
        # writer could insert a third dup or shift salience between the two
        # lock acquisitions. Merge into one with-block.
        downweighted = 0
        with self.store._lock:
            groups = self.store.conn.execute(
                "SELECT content, GROUP_CONCAT(id) FROM memories "
                "WHERE content IS NOT NULL "
                "  AND (kind IS NULL OR kind != 'entity') "
                "GROUP BY content HAVING COUNT(*) > 1"
            ).fetchall()
            for _content, ids_csv in groups:
                ids = [int(x) for x in ids_csv.split(',')]
                sal_rows = self.store.conn.execute(
                    f"SELECT id, salience FROM memories "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    tuple(ids),
                ).fetchall()
                max_id = max(sal_rows, key=lambda r: r[1] or 0)[0]
                for mid, _sal in sal_rows:
                    if mid == max_id:
                        continue
                    self.store.conn.execute(
                        "UPDATE memories SET salience = salience * ? WHERE id = ?",
                        (decay_factor, mid),
                    )
                    downweighted += 1
            self.store.conn.commit()
        return {"duplicates_downweighted": downweighted}

    def create_insight_from_cluster(self, memory_ids: list[int]) -> int:
        """Create a dream_insight node summarizing the cluster + add
        summarizes edges back to source memories (evidence-attached).

        Per handoff sec 9.3: dream insights MUST have evidence edges,
        never free-floating.
        """
        if not memory_ids:
            return 0
        with self.store._lock:
            rows = self.store.conn.execute(
                f"SELECT content FROM memories "
                f"WHERE id IN ({','.join('?' * len(memory_ids))})",
                tuple(memory_ids),
            ).fetchall()
        contents = [r[0] or "" for r in rows]
        summary = "Insight from cluster: " + " | ".join(c[:80] for c in contents[:3])
        insight_id = self.remember(
            summary,
            label="dream-insight",
            detect_conflicts=False,
            kind="dream_insight",
            origin_system="dream_engine",
        )
        for mid in memory_ids:
            try:
                self.store.add_connection(
                    insight_id, mid, weight=1.0, edge_type="summarizes",
                )
            except Exception:
                pass
        return insight_id

    @classmethod
    def _content_jaccard(cls, a: str, b: str) -> float:
        """Word-level Jaccard similarity, lowercased + stopword-filtered."""
        if not a or not b:
            return 0.0
        wa = {w for w in a.lower().split() if w not in cls._CONTRADICTION_STOPWORDS}
        wb = {w for w in b.lower().split() if w not in cls._CONTRADICTION_STOPWORDS}
        if not wa or not wb:
            return 0.0
        inter = wa & wb
        union = wa | wb
        return len(inter) / len(union)

    # ------------------------------------------------------------------
    # Phase 7 Commit 10: locus overlay + governance + explanation paths
    # ------------------------------------------------------------------

    def create_locus(self, wing: str, room: str) -> int:
        """Create a locus_room node and link it under a locus_wing parent
        (auto-creating the wing if missing). Returns the room locus id.

        Loci are kind='locus' nodes in the unified graph; located_in edges
        connect memories -> rooms -> wings. Per handoff sec 4.2, locus is an
        OVERLAY, not the canonical durable structure.
        """
        wing_id = self._get_or_create_locus_node(wing, level="wing", parent=None)
        room_id = self._get_or_create_locus_node(room, level="room", parent=wing_id)
        return room_id

    def _get_or_create_locus_node(self, label: str, *, level: str,
                                   parent: Optional[int]) -> int:
        """Find or create a locus node with the given label + level."""
        with self.store._lock:
            row = self.store.conn.execute(
                "SELECT id FROM memories WHERE kind = 'locus' AND label = ? LIMIT 1",
                (label,),
            ).fetchone()
        if row:
            return row[0]
        meta = {"level": level, "parent": parent}
        node_id = self.store.store(
            label=label,
            content=f"Locus {level}: {label}",
            embedding=[0.0] * 16,
            kind="locus",
            origin_system="locus_overlay",
            metadata=meta,
        )
        if parent is not None:
            try:
                self.store.add_connection(node_id, parent, weight=1.0,
                                          edge_type="located_in")
            except Exception:
                pass
        return node_id

    def assign_locus(self, memory_id: int, locus_id: int) -> None:
        """Add a located_in edge from memory to locus. Idempotent."""
        if self.has_edge(memory_id, locus_id, edge_type="located_in"):
            return
        self.store.add_connection(memory_id, locus_id, weight=1.0,
                                  edge_type="located_in")

    def memory_count(self, *, exclude_overlay: bool = True) -> int:
        """Count user memories. By default excludes entity + locus overlay
        nodes (which are derived/system, not user-authored memories)."""
        with self.store._lock:
            if exclude_overlay:
                row = self.store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE kind IS NULL "
                    "OR (kind != 'entity' AND kind != 'locus')"
                ).fetchone()
            else:
                row = self.store.conn.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()
        return row[0]

    def forget(self, memory_id: int, *, mode: str = "background") -> None:
        """Soft-forget operation per handoff sec 11.4.

        Modes:
            'background': sets memory_visibility='backgrounded' — keeps the
                row + edges intact, just deprioritizes from default recall.
            'delete':     hard-delete (rare; use 'background' first).
            'redact':     replace content with a redaction marker; preserves
                edges + audit trail per H19 invariant.

        Default 'background' is the safe choice.
        """
        if mode == "background":
            with self.store._lock:
                self.store.conn.execute(
                    "UPDATE memories SET memory_visibility = 'backgrounded' "
                    "WHERE id = ?",
                    (memory_id,),
                )
                self.store.conn.commit()
        elif mode == "delete":
            # Per Reviewer #1: connections has FOREIGN KEY but no CASCADE
            # and PRAGMA foreign_keys is never set ON. Hard-delete leaves
            # dangling edges. Cascade manually within the same lock window.
            with self.store._lock:
                self.store.conn.execute(
                    "DELETE FROM connections "
                    "WHERE source_id = ? OR target_id = ?",
                    (memory_id, memory_id),
                )
                self.store.conn.execute(
                    "DELETE FROM memories WHERE id = ?", (memory_id,),
                )
                # Also clean FTS index (silent no-op if FTS5 unavailable).
                try:
                    self.store.conn.execute(
                        "DELETE FROM memories_fts WHERE rowid = ?", (memory_id,),
                    )
                except sqlite3.OperationalError:
                    pass
                self.store.conn.commit()
        elif mode == "redact":
            with self.store._lock:
                self.store.conn.execute(
                    "UPDATE memories SET content = '[REDACTED]', "
                    "memory_visibility = 'hidden' WHERE id = ?",
                    (memory_id,),
                )
                self.store.conn.commit()
        else:
            raise ValueError(f"unknown forget mode: {mode!r}")

    def explain_recall(self, query: str, k: int = 5,
                       *, kind: Optional[str] = None,
                       as_of: Optional[float] = None) -> list[dict[str, Any]]:
        """recall() with per-result explanation paths attached.

        Returns the same dicts as recall(), each augmented with an
        `explanation` key containing:
            - channels: list of channels that contributed
            - final_score: combined score
            - features: dict of feature scores including 'salience'

        Per addendum lines 541-547 + handoff sec 12.4.
        """
        results = self.recall(query, k=k, kind=kind, as_of=as_of)
        intent = self._classify_intent(query)
        for r in results:
            r["explanation"] = {
                "query": query,
                "intent": intent,
                "channels": ["dense"]
                            + (["sparse"] if self.sparse_search else [])
                            + (["temporal"] if as_of is not None else [])
                            + (["graph"] if self._graph_nodes else []),
                "final_score": r.get("combined", r.get("similarity", 0.0)),
                "features": {
                    "semantic":          r.get("similarity", 0.0),
                    "temporal_score":    r.get("temporal_score", 0.0),
                    "salience":          r.get("salience_factor", 1.0),
                    "combined":          r.get("combined", 0.0),
                },
            }
        return results

    def run_contradiction_detection_once(self,
                                          jaccard_threshold: float = 0.4) -> dict[str, int]:
        """Detect pairs where one memory's validity ends before another's
        begins AND they have substantial content overlap. Adds contradicts
        edges (does NOT delete or modify memories — H6/H19 invariant).

        Per addendum lines 498-503 + handoff sec 9.4.
        """
        with self.store._lock:
            rows = self.store.conn.execute(
                "SELECT id, content, valid_from, valid_to FROM memories "
                "WHERE (kind IS NULL OR kind != 'entity') "
                "  AND content IS NOT NULL"
            ).fetchall()
        edges_added = 0
        # Compare pairs where one has a valid_to and another has valid_from
        # strictly greater. O(n^2) but acceptable for AE scale (1k-10k memories).
        items = [(r[0], r[1] or "", r[2], r[3]) for r in rows]
        for old in items:
            old_id, old_content, _ovf, ovt = old
            if ovt is None:
                continue
            for new in items:
                new_id, new_content, nvf, _nvt = new
                if new_id == old_id or nvf is None:
                    continue
                if nvf <= ovt:
                    continue
                if self._content_jaccard(old_content, new_content) < jaccard_threshold:
                    continue
                if self.has_edge(old_id, new_id, edge_type="contradicts"):
                    continue
                try:
                    self.store.add_connection(
                        old_id, new_id, weight=1.0, edge_type="contradicts",
                    )
                    edges_added += 1
                except Exception:
                    pass
        return {"contradiction_edges_added": edges_added}

    def _filter_by_kind(self, results: list[dict], kind: str) -> list[dict]:
        """Filter recall results to only memories whose `kind` column matches.

        Single batched SELECT — one DB roundtrip regardless of result-set size.
        """
        if not results:
            return []
        ids = [r['id'] for r in results]
        placeholders = ",".join("?" * len(ids))
        with self.store._lock:
            rows = self.store.conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) AND kind = ?",
                (*ids, kind),
            ).fetchall()
        matching = {row[0] for row in rows}
        return [r for r in results if r['id'] in matching]

    def _recall_inner(self, query: str, k: int = 5, temporal_weight: float = 0.2) -> list[dict]:
        """Original recall implementation — used by recall() and Phase 7
        retrieval channels. Maintains pre-Phase-7 behavior when called with
        no extra args.
        """
        import math
        import time

        query_vec = self.embedder.embed(query)
        now = time.time()

        # C++ fast path: SIMD retrieve returns top-k candidates in microseconds
        # Then apply temporal scoring on the small candidate set
        if self._cpp:
            try:
                candidates = self._cpp.retrieve(query_vec, k=k * 3)
                if candidates:
                    # Batch-fetch salience meta for the candidate set so the
                    # salience factor can be applied without re-reading embeddings.
                    cand_ids = [self._cpp_id_map.get(c['id'], c['id']) for c in candidates]
                    meta = self.store.get_meta_many(cand_ids) if hasattr(self.store, 'get_meta_many') else {}

                    scored = []
                    for c in candidates:
                        cpp_id = c['id']
                        # Map C++ index back to SQLite ID
                        mem_id = self._cpp_id_map.get(cpp_id, cpp_id)
                        sim = c.get('similarity', c.get('score', 0))
                        node = self._graph_nodes.get(mem_id, {})
                        m = meta.get(mem_id, {})
                        salience_factor = self._effective_salience(
                            m.get('salience'), m.get('access_count'), m.get('created_at'), now=now
                        )

                        base_combined = (1 - temporal_weight) * sim + temporal_weight * 0.5
                        # H5: salience multiply gated by self._salience_multiply
                        effective_sal = salience_factor if self._salience_multiply else 1.0
                        scored.append({
                            'id': mem_id,
                            'label': c.get('label', node.get('label', '')),
                            'content': c.get('content', ''),
                            'embedding': node.get('embedding', []),
                            'similarity': sim,
                            'temporal_score': 0.5,
                            'salience_factor': round(salience_factor, 4),
                            'combined': base_combined * effective_sal,
                            'connections': list(node.get('connections', {}).keys()),
                        })

                    scored.sort(key=lambda x: -x['combined'])
                    # Optional cross-encoder rerank on the candidate set before slicing
                    scored = self._maybe_rerank(query, scored)
                    # Touch accessed memories
                    for s in scored[:k]:
                        try:
                            self.store.touch(s['id'])
                        except Exception:
                            pass
                    return scored[:k]
            except Exception:
                pass  # Fall through to Python path

        # Python path: prefer HNSW ANN over brute force when available
        if self._hnsw is not None and self._hnsw_count > 0:
            try:
                import numpy as np
                q = np.asarray([query_vec], dtype=np.float32)
                fetch = max(k * 3, min(50, self._hnsw_count))
                labels, dists = self._hnsw.knn_query(q, k=min(fetch, self._hnsw_count))
                ids = [int(x) for x in labels[0]]
                sims = [float(1.0 - d) for d in dists[0]]
                meta_map = self.store.get_meta_many(ids) if hasattr(self.store, 'get_meta_many') else {}
                scored = []
                for cid, sim in zip(ids, sims):
                    mem = self.store.get(cid)
                    if not mem:
                        continue
                    row = self.store.conn.execute(
                        "SELECT last_accessed FROM memories WHERE id = ?", (cid,)
                    ).fetchone()
                    import math as _m
                    if row and row[0]:
                        age_hours = (now - row[0]) / 3600
                        temporal_score = _m.exp(-0.693 * age_hours / 24)
                    else:
                        temporal_score = 0.5
                    m = meta_map.get(cid, {})
                    sal = self._effective_salience(
                        m.get('salience'), m.get('access_count'), m.get('created_at'), now=now
                    )
                    base = (1 - temporal_weight) * sim + temporal_weight * temporal_score
                    # H5: salience multiply gated
                    effective_sal = sal if self._salience_multiply else 1.0
                    scored.append({
                        **mem,
                        'similarity': sim,
                        'temporal_score': temporal_score,
                        'salience_factor': sal,
                        'combined': base * effective_sal,
                    })
                scored.sort(key=lambda x: -x['combined'])
                rerank_pool = scored[:max(k * 3, 15)] if self._rerank else scored
                rerank_pool = self._maybe_rerank(query, rerank_pool)

                results = []
                seen = set()
                for mem in rerank_pool[:k]:
                    if mem['id'] in seen:
                        continue
                    seen.add(mem['id'])
                    conns = self.store.get_connections(mem['id'])
                    connected = []
                    for c in conns:
                        other = c['target'] if c['source'] == mem['id'] else c['source']
                        if other not in seen:
                            other_mem = self.store.get(other)
                            if other_mem:
                                connected.append({
                                    'id': other, 'label': other_mem['label'], 'weight': c['weight']
                                })
                    results.append({
                        'id': mem['id'],
                        'label': mem['label'],
                        'content': mem['content'],
                        'similarity': round(mem['similarity'], 4),
                        'temporal_score': round(mem['temporal_score'], 4),
                        'salience_factor': round(mem.get('salience_factor', 1.0), 4),
                        'combined': round(mem['combined'], 4),
                        'connections': connected[:3],
                    })
                    self.store.touch(mem['id'])
                if results:
                    return results
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("hnsw recall failed, falling back to brute force: %s", exc)

        # Python brute-force fallback: O(n) linear scan
        scored = []
        for mem in self.store.get_all():
            sim = self._cosine_similarity(query_vec, mem['embedding'])

            # Temporal score: exponential decay based on last_accessed
            try:
                row = self.store.conn.execute(
                    "SELECT last_accessed FROM memories WHERE id = ?", (mem['id'],)
                ).fetchone()
                if row and row[0]:
                    age_hours = (now - row[0]) / 3600
                    temporal_score = math.exp(-0.693 * age_hours / 24)
                else:
                    temporal_score = 0.5
            except Exception:
                temporal_score = 0.5

            salience_factor = self._effective_salience(
                mem.get('salience'), mem.get('access_count'), mem.get('created_at'), now=now
            )
            base_combined = (1 - temporal_weight) * sim + temporal_weight * temporal_score
            # H5: salience multiply gated
            effective_sal = salience_factor if self._salience_multiply else 1.0
            combined = base_combined * effective_sal
            scored.append({
                **mem,
                'similarity': sim,
                'temporal_score': temporal_score,
                'salience_factor': salience_factor,
                'combined': combined,
            })

        # Sort by combined score
        scored.sort(key=lambda x: -x['combined'])

        # Optional cross-encoder rerank on the top candidates before slicing to k
        rerank_pool = scored[:max(k * 3, 15)] if self._rerank else scored
        rerank_pool = self._maybe_rerank(query, rerank_pool)

        # Enrich with connections via spreading activation
        results = []
        seen = set()
        for mem in rerank_pool[:k]:
            if mem['id'] in seen:
                continue
            seen.add(mem['id'])
            
            # Get connections
            conns = self.store.get_connections(mem['id'])
            connected = []
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                if other not in seen:
                    other_mem = self.store.get(other)
                    if other_mem:
                        connected.append({
                            'id': other,
                            'label': other_mem['label'],
                            'weight': c['weight']
                        })
            
            results.append({
                'id': mem['id'],
                'label': mem['label'],
                'content': mem['content'],
                'similarity': round(mem['similarity'], 4),
                'temporal_score': round(mem['temporal_score'], 4),
                'salience_factor': round(mem.get('salience_factor', 1.0), 4),
                'combined': round(mem['combined'], 4),
                'connections': connected[:3],  # Top 3
            })
            
            self.store.touch(mem['id'])
        
        return results
    
    def think(self, start_id: int, depth: int = 3, decay: float = 0.85,
              engine: str = "bfs", alpha: float = 0.15, n_iter: int = 40,
              top_k: int = 25) -> list[dict]:
        """
        Spreading activation from a starting memory.

        engine='bfs'  — original decay-BFS propagation (default; preserves prior behavior)
        engine='ppr'  — Personalized PageRank seeded at start_id (HippoRAG-2 style).
                        Uses weighted edges and restart prob `alpha`. Tends to rank
                        globally-relevant associates higher than local neighbors,
                        and is less sensitive to `depth` or hop-count heuristics.

        Returns activated memories sorted by activation/score, limited to top_k.
        """
        # In lazy_graph mode, pull the seed on demand so the early-return gate
        # doesn't short-circuit before any nodes have been cached.
        if start_id not in self._graph_nodes:
            if self._lazy_graph:
                if not self._ensure_node(start_id):
                    return []
            else:
                return []

        if engine == "ppr":
            activation = self._ppr([start_id], alpha=alpha, n_iter=n_iter)
        else:
            activation = self._bfs_spread(start_id, depth=depth, decay=decay)

        results = []
        for node_id, act in activation.items():
            if node_id == start_id:
                continue
            mem = self.store.get(node_id)
            if mem:
                results.append({
                    'id': node_id,
                    'label': mem['label'],
                    'activation': round(act, 6),
                })

        results.sort(key=lambda x: -x['activation'])
        return results[:top_k]

    def _bfs_spread(self, start_id: int, depth: int = 3, decay: float = 0.85) -> dict:
        """Original decay-BFS spreading activation — preserved as a fallback engine."""
        activation = {start_id: 1.0}
        visited = {start_id}
        queue = [(start_id, 1.0, 0)]

        while queue:
            current, act, level = queue.pop(0)
            if level >= depth or act < 0.01:
                continue

            # In lazy mode, hydrate the node on first visit so the walk can continue
            if self._lazy_graph and current not in self._graph_nodes:
                self._ensure_node(current)
            node = self._graph_nodes.get(current, {})
            for neighbor_id, weight in node.get('connections', {}).items():
                propagated = act * weight * decay
                if propagated < 0.01:
                    continue

                if neighbor_id not in activation or propagated > activation[neighbor_id]:
                    activation[neighbor_id] = propagated
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, propagated, level + 1))

        return activation

    def _ppr(self, seed_ids, alpha: float = 0.15, n_iter: int = 40) -> dict:
        """Personalized PageRank over the in-memory weighted graph.

        Based on HippoRAG 2's retrieval primitive: `alpha` is the restart
        probability (fraction of mass that returns to the seed distribution
        at each iteration). Equivalent to hippocampal indexing via PPR on a
        phrase/passage graph — same philosophy as spreading activation, but
        with a principled convergence criterion and no depth cliff.

        Uses networkx when available (weighted PR); falls back to pure-numpy
        power iteration for offline/minimal-dep environments.
        """
        # In lazy mode, ensure seeds and their reachable subgraph are loaded.
        # PPR needs the full structure it walks over; we hydrate via a bounded
        # BFS expansion so we don't blow memory on huge graphs.
        if self._lazy_graph:
            for s in seed_ids:
                self._ensure_node(s)
            # Two-hop neighborhood expansion (bounded to avoid O(|V|) load)
            frontier = set(seed_ids)
            for _ in range(2):
                next_frontier = set()
                for nid in frontier:
                    node = self._graph_nodes.get(nid, {})
                    for nb in node.get('connections', {}):
                        if nb not in self._graph_nodes:
                            self._ensure_node(nb)
                            next_frontier.add(nb)
                frontier = next_frontier

        # Personalization vector concentrated on seeds
        seeds = [s for s in seed_ids if s in self._graph_nodes]
        if not seeds:
            return {}

        # Prefer networkx's pagerank (handles dangling, weights, convergence)
        try:
            import networkx as nx  # type: ignore
            g = nx.DiGraph()
            for nid, node in self._graph_nodes.items():
                g.add_node(nid)
                for nb, w in node.get('connections', {}).items():
                    if w and w > 0:
                        g.add_edge(nid, nb, weight=float(w))
            if g.number_of_edges() == 0:
                return {s: 1.0 / len(seeds) for s in seeds}
            pers = {n: (1.0 / len(seeds) if n in seeds else 0.0) for n in g.nodes}
            pr = nx.pagerank(g, alpha=1.0 - alpha, personalization=pers,
                             weight='weight', max_iter=n_iter, tol=1e-6)
            return pr
        except Exception:
            pass

        # Pure-Python power iteration fallback (O(iter * |E|))
        nodes = list(self._graph_nodes.keys())
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        if n == 0:
            return {}

        # Weighted out-degree normalized transition
        out_rows: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        out_totals = [0.0] * n
        for src, node in self._graph_nodes.items():
            i = idx[src]
            for tgt, w in node.get('connections', {}).items():
                if tgt in idx and w and w > 0:
                    out_rows[i].append((idx[tgt], float(w)))
                    out_totals[i] += float(w)

        # Personalization: uniform over seeds
        p = [0.0] * n
        for s in seeds:
            p[idx[s]] = 1.0 / len(seeds)

        r = list(p)
        damp = 1.0 - alpha
        for _ in range(n_iter):
            new = [alpha * p_i for p_i in p]
            for i in range(n):
                total = out_totals[i]
                if total <= 0:
                    # Dangling node — redistribute mass uniformly to seeds
                    for s in seeds:
                        new[idx[s]] += damp * r[i] / len(seeds)
                    continue
                share = damp * r[i] / total
                for j, w in out_rows[i]:
                    new[j] += share * w
            # Normalize against drift
            s_new = sum(new) or 1.0
            r = [v / s_new for v in new]

        return {nodes[i]: r[i] for i in range(n)}
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get all connections for a memory."""
        conns = self.store.get_connections(mem_id)
        results = []
        for c in conns:
            other = c['target'] if c['source'] == mem_id else c['source']
            mem = self.store.get(other)
            if mem:
                results.append({
                    'id': other,
                    'label': mem['label'],
                    'weight': round(c['weight'], 4),
                    'type': c['type']
                })
        return results
    
    def graph(self) -> dict:
        """Get knowledge graph stats and structure."""
        stats = self.store.get_stats()
        
        # Build adjacency summary
        edges = []
        seen = set()
        for node_id, node in self._graph_nodes.items():
            for other_id, weight in node.get('connections', {}).items():
                key = tuple(sorted([node_id, other_id]))
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        'from': node_id,
                        'to': other_id,
                        'weight': round(weight, 3)
                    })
        
        return {
            'nodes': stats['memories'],
            'edges': len(edges),
            'top_edges': sorted(edges, key=lambda x: -x['weight'])[:10]
        }
    
    def stats(self) -> dict:
        """Get memory statistics including Phase B feature state.

        H7: expanded to report feature availability so `mem.stats()` answers
        'is HNSW on? is rerank loaded? is networkx/Louvain available?' at one call.
        """
        graph = self.store.stats() if hasattr(self.store, 'stats') else self.store.get_stats()
        try:
            import networkx  # noqa: F401
            nx_ok = True
        except ImportError:
            nx_ok = False
        return {
            'memories': graph['memories'],
            'connections': graph['connections'],
            'embedding_dim': self.dim,
            'embedding_backend': self.embedder.backend.__class__.__name__,
            # Phase B feature flags
            'cpp_available': self._cpp is not None,
            'hnsw_active': self._hnsw is not None,
            'hnsw_count': self._hnsw_count if self._hnsw is not None else 0,
            'hnsw_ef': getattr(self, '_hnsw_ef', None),
            'lazy_graph': self._lazy_graph,
            'louvain_available': nx_ok,
            'reranker_loaded': self._rerank_model is not None,
            'rerank_enabled': self._rerank,
            'salience_multiply': self._salience_multiply,
        }
    
    def close(self):
        # H4: flush HNSW before closing so next session starts fast
        self._save_hnsw()
        if self._cpp:
            try:
                self._cpp.shutdown()
            except Exception:
                pass
            self._cpp = None
        self.store.close()
    
    # Cython-accelerated ops (falls back to Python if unavailable)
    try:
        from fast_ops import cosine_similarity as _cosine_sim_fast
    except ImportError:
        _cosine_sim_fast = None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if NeuralMemory._cosine_sim_fast is not None:
            import numpy as np
            # Avoid repeated array creation for lists
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b, dtype=np.float64)
            return float(NeuralMemory._cosine_sim_fast(a, b))
        dot = sum(x*y for x, y in zip(a, b))
        na = (sum(x*x for x in a)) ** 0.5
        nb = (sum(x*x for x in b)) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
