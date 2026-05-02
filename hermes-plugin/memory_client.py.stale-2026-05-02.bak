#!/usr/bin/env python3
"""
memory_client.py - Python client for Neural Memory Adapter
Wraps the C++ library via ctypes. Uses embed_provider for text->vector.
"""

import ctypes
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

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
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        self.conn.execute("PRAGMA synchronous=NORMAL") # Faster writes
        self.conn.executescript(SCHEMA)
        _migrate_bitemporal(self.conn)
        self.conn.commit()
        self._lock = threading.Lock()
    
    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                (label, content, blob)
            )
            self.conn.commit()
            return cur.lastrowid
    
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
                       embedding: list[float]) -> None:
        """H19: cleanly replace a memory's content + label + embedding in place.
        No `[SUPERSEDED]` prefix. Caller should have called archive_superseded()
        first to preserve audit history.
        """
        blob = struct.pack(f'{len(embedding)}f', *embedding) if embedding else None
        with self._lock:
            self.conn.execute(
                """UPDATE memories SET content = ?, label = ?, embedding = ?
                   WHERE id = ?""",
                (content, label, blob, memory_id),
            )
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
        mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn_count = self.conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        return {'memories': mem_count, 'connections': conn_count}
    
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
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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

        self.dim = self.embedder.dim

        # Cross-encoder reranker (opt-in, lazy-loaded).
        # Keeps default behavior identical when rerank=False.
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
                 normalize_dates: bool = True) -> int:
        """Store a memory. Returns memory ID.

        If detect_conflicts=True, checks for existing memories about the same
        topic that contain contradictory information and updates/invalidates them.

        If normalize_dates=True (default; H18), relative dates in the content
        are converted to absolute ISO dates before storage. Pass False to
        preserve verbatim text.
        """
        import math
        import time

        if normalize_dates:
            text = self._normalize_dates(text)

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
                        self.store.replace_memory(
                            memory_id=conflict_id,
                            content=text,
                            label=label or text[:60],
                            embedding=embedding,
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
        
        mem_id = self.store.store(label or text[:60], text, embedding)
        
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
    
    def _maybe_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder rerank of a candidate set. No-op if disabled or model unavailable.

        Candidates must carry 'content' (string). Preserves all other fields and
        adds 'rerank_score'. Re-sorts by rerank_score descending.
        """
        if not self._rerank or not candidates:
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

    def recall(self, query: str, k: int = 5, temporal_weight: float = 0.2) -> list[dict]:
        """
        Retrieve memories related to query.

        Args:
            query: Search query
            k: Number of results
            temporal_weight: Weight for recency scoring (0=pure similarity, 1=pure recency)

        Returns list of {id, label, content, similarity, temporal_score, connections}.
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
