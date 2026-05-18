#!/usr/bin/env python3
"""
live_server.py — FastAPI WebSocket server for Mazemaker Live Dashboard.

Reads from the SQLite source-of-truth.

Usage:
    python live_server.py                             # auto-detect DB
    python live_server.py --port 8443                 # custom port
    python live_server.py --db /path/to/memory.db     # custom SQLite path
    python live_server.py --no-tls                    # HTTP only
    python live_server.py --desktop-layer             # reduced logging
    python live_server.py --watch-interval 1          # poll every 1s
"""
import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# pty/termios/fcntl are POSIX-only and only used for the embedded
# terminal handler. Import lazily so the dashboard at least boots on
# Windows for read-only views; routes that need a PTY fail gracefully.
try:
    import fcntl  # noqa: F401  POSIX
    import pty  # noqa: F401   POSIX
    import termios  # noqa: F401  POSIX
    _HAS_PTY = True
except ImportError:
    _HAS_PTY = False

logger = logging.getLogger("neural-live")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

def _default_sqlite_path() -> str:
    """Resolve the engine's SQLite store path.  The repo was renamed
    from `neural-memory` → `mazemaker` (2026-05-01), and the on-disk
    layout moved with it.  Try the new location first, fall back to
    the legacy one so older installs still get picked up.
    """
    candidates = (
        "~/.mazemaker/data/memory.db",   # current (post-rename)
        "~/.mazemaker/memory.db",        # transitional
        "~/.neural_memory/memory.db",    # legacy
    )
    for p in candidates:
        full = os.path.expanduser(p)
        if os.path.exists(full):
            return full
    return os.path.expanduser(candidates[0])  # default for empty installs

DEFAULT_SQLITE = _default_sqlite_path()
TEMPLATE_DIR   = Path(__file__).parent
POLL_INTERVAL  = 2.0
CONFIG_PATH    = os.path.expanduser("~/.hermes/config.yaml")
NODE_LIMIT     = int(os.environ.get("MM_DASH_NODE_LIMIT", "2000"))   # top hub nodes
EDGE_LIMIT     = int(os.environ.get("MM_DASH_EDGE_LIMIT", "8000"))   # max edges

_DB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-worker")

# Per-process TTL cache for slow MSSQL aggregations. NeuralMemory has 696k
# rows; the metadata_json LIKE-GROUP-BY scans them all (~1.4s) and the
# distribution barely shifts between polls. Caching 60s drops the steady
# state to ~150ms.
import threading as _threading
_mssql_cache: dict = {}
_mssql_cache_lock = _threading.Lock()


def _cache_get(key: str, ttl: float):
    with _mssql_cache_lock:
        entry = _mssql_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            return None
        return value


def _cache_set(key: str, value, ttl: float) -> None:
    with _mssql_cache_lock:
        _mssql_cache[key] = (time.time() + ttl, value)


CACHE_TTL_CATS = 60.0      # categories — drift slowly
CACHE_TTL_DIM  = 24*3600   # embedding_dim — only changes on model swap
CACHE_TTL_NM_COUNT = 30.0  # NeuralMemory total count — bulk-archive, slow growth

_metrics: dict = {
    "start_time":        time.time(),
    "db_query_ms":       0.0,
    "broadcast_ms":      0.0,
    "updates_sent":      0,
    "ws_total":          0,
    "last_update_time":  0.0,
    "poll_count":        0,
    "db_source":         "unknown",
}

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    print("FastAPI + uvicorn required: pip install fastapi uvicorn")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Library bootstrap (xterm.js)
# ═══════════════════════════════════════════════════════════

_XTERM_LIBS = [
    ("xterm-4.19.0.min.js",        "https://cdn.jsdelivr.net/npm/xterm@4.19.0/lib/xterm.min.js"),
    ("xterm-4.19.0.css",           "https://cdn.jsdelivr.net/npm/xterm@4.19.0/css/xterm.css"),
    ("xterm-addon-fit-0.5.0.min.js",
     "https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.5.0/lib/xterm-addon-fit.min.js"),
]


def _ensure_libs() -> None:
    libs_dir = TEMPLATE_DIR / ".lib_cache"
    libs_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in _XTERM_LIBS:
        dest = libs_dir / filename
        if dest.exists():
            continue
        logger.info(f"Downloading {filename} …")
        try:
            urllib.request.urlretrieve(url, str(dest))
            kb = dest.stat().st_size // 1024
            logger.info(f"  ✓ {filename} ({kb} KB)")
        except Exception as exc:
            logger.warning(f"  ✗ {filename}: {exc}")


# ═══════════════════════════════════════════════════════════
# Categorizer
# ═══════════════════════════════════════════════════════════

def _categorize(label: str) -> str:
    if label.startswith("peer:"):              return "Peer"
    if label.startswith(("turn:", "msg:")):    return "Conversation"
    if label.startswith("session:"):           return "Session"
    if label.startswith("doc:"):               return "Document"
    if label.startswith("skill:"):             return "Skill"
    return "Other"


# ═══════════════════════════════════════════════════════════
# Agent classifier — for the architect mirror.
#
# Parses memory labels to figure out which agent ingested the row.
# Labels are written by the various MCP clients and the dream engine
# itself.  Pattern is `auto:<agent>:<session>:t<N>:[u|a]` for
# conversation turns, plus a few well-known prefixes for synthesised
# nodes (derived: clusters, peer: handles).
#
# Color palette is the brand accent ladder so the mirror reads as a
# Mazemaker artefact, not a generic ops dashboard.
# ═══════════════════════════════════════════════════════════

AGENT_PALETTE = {
    "Claude":   "#76d9ff",   # bright cyan — the loudest co-author
    "Hermes":   "#ffd700",   # warm gold
    "GPT":      "#34d399",   # mint green
    "Gemini":   "#60a5fa",   # blue
    "Mistral":  "#f97316",   # orange
    "Qwen":     "#ec4899",   # pink
    "Gemma":    "#a78bfa",   # violet
    "User":     "#ededf2",   # near-white
    "Dream":    "#bf00ff",   # mazemaker accent — synthesised by sleep
    "Peer":     "#ccff00",   # neon green (matches existing peer color)
    "Other":    "#5e5e6b",   # dim — the unclassified background
}


def _classify_agent(label: str) -> str:
    """Best-effort agent attribution from a memory label.

    Tightly coupled to the MCP plugins' label-writing convention
    (`auto:<agent>:<session>:t<N>:<role>`).  Unknown labels collapse
    to "Other" so the mirror keeps rendering them at lower visual
    weight.
    """
    if not label:
        return "Other"
    lo = label.lower()
    if lo.startswith("derived:"):
        return "Dream"
    if lo.startswith("peer:"):
        return "Peer"
    if lo.startswith(("auto:claude", "claude:", "claude-turn")):
        return "Claude"
    if lo.startswith(("auto:hermes", "hermes:")):
        return "Hermes"
    if lo.startswith(("auto:gpt", "gpt:", "auto:openai", "openai:")):
        return "GPT"
    if lo.startswith(("auto:gemini", "gemini:")):
        return "Gemini"
    if lo.startswith(("auto:mistral", "mistral:", "auto:ministral")):
        return "Mistral"
    if lo.startswith(("auto:qwen", "qwen:")):
        return "Qwen"
    if lo.startswith(("auto:gemma", "gemma:")):
        return "Gemma"
    if lo.startswith(("auto:user", "user:")) or lo.endswith(":u"):
        return "User"
    return "Other"


def _agent_color(agent: str) -> str:
    return AGENT_PALETTE.get(agent, AGENT_PALETTE["Other"])


# ═══════════════════════════════════════════════════════════
# Data readers
# ═══════════════════════════════════════════════════════════

def read_sqlite(db_path: str) -> dict:
    import sqlite3
    t0 = time.perf_counter()
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        return _read_sqlite_body(conn, t0, db_path)
    finally:
        conn.close()


def _read_sqlite_body(conn, t0, db_path):
    cur = conn.cursor()

    cur.execute("""
        SELECT m.id, m.label, m.content, m.salience, m.access_count, m.created_at,
               COALESCE(out_d.out_degree, 0),
               COALESCE(in_d.in_degree,  0),
               COALESCE(out_d.avg_weight, 0)
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree, AVG(weight) AS avg_weight
                   FROM connections GROUP BY source_id) out_d ON m.id = out_d.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) in_d ON m.id = in_d.target_id
        ORDER BY (COALESCE(out_d.out_degree,0) + COALESCE(in_d.in_degree,0)) DESC
    """)
    nodes = []
    for r in cur.fetchall():
        lbl = r[1] or ""
        nodes.append({
            "id": r[0], "label": lbl[:50], "category": _categorize(lbl),
            "content_length": len(r[2]) if r[2] else 0,
            "salience": r[3] or 1.0, "access_count": r[4] or 0,
            "created_at": r[5] or 0,
            "out_degree": r[6], "in_degree": r[7], "total_degree": r[6] + r[7],
            "avg_weight": round(r[8], 4),
        })

    hub_ids = [n["id"] for n in nodes[:NODE_LIMIT]]
    if hub_ids:
        # `IN (?, ...)` with bound params is safer than f-stringing ints, and
        # also avoids the empty-DB \"IN ()\" syntax error that would crash
        # the dashboard's first paint on a freshly-installed empty store.
        placeholders = ",".join("?" * len(hub_ids))
        cur.execute(
            f"SELECT source_id, target_id, weight FROM connections "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) "
            f"ORDER BY weight DESC LIMIT {int(EDGE_LIMIT)}",
            hub_ids + hub_ids,
        )
        edges = [{"source": r[0], "target": r[1], "weight": round(r[2], 4)} for r in cur.fetchall()]
    else:
        edges = []

    cur.execute("""
        SELECT cat, COUNT(*) FROM (
            SELECT CASE
                WHEN label LIKE 'peer:%'                       THEN 'Peer'
                WHEN label LIKE 'turn:%' OR label LIKE 'msg:%' THEN 'Conversation'
                WHEN label LIKE 'session:%'                    THEN 'Session'
                WHEN label LIKE 'doc:%'                        THEN 'Document'
                WHEN label LIKE 'skill:%'                      THEN 'Skill'
                ELSE 'Other'
            END AS cat FROM memories
        ) GROUP BY cat ORDER BY COUNT(*) DESC
    """)
    categories = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("""
        SELECT bucket, COUNT(*) FROM (
            SELECT CASE
                WHEN weight >= 0.8 THEN 'Strong (0.8-1.0)'
                WHEN weight >= 0.6 THEN 'Med-Strong (0.6-0.8)'
                WHEN weight >= 0.4 THEN 'Medium (0.4-0.6)'
                WHEN weight >= 0.2 THEN 'Weak (0.2-0.4)'
                ELSE 'Very Weak (0-0.2)'
            END AS bucket FROM connections
        ) GROUP BY bucket
    """)
    weights = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) FROM memories");    n_mem  = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections"); n_conn = cur.fetchone()[0]

    cur.execute("SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1")
    emb_row    = cur.fetchone()
    actual_dim = (emb_row[0] // 4) if emb_row else 1024

    dream_sessions: list[dict] = []
    try:
        cur.execute(
            "SELECT id, phase, started_at, completed_at, stats "
            "FROM dream_sessions ORDER BY started_at"
        )
        for r in cur.fetchall():
            dream_sessions.append({
                "id": r[0], "phase": r[1],
                "started_at": r[2], "completed_at": r[3], "stats": r[4],
            })
    except Exception:
        pass

    ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["db_query_ms"] = ms
    # Trim node list to the visible top-N hubs.  The SQL above orders by
    # total_degree DESC so this keeps the strongest hubs.  Without this
    # cap the bootstrap HTML inlined every memory in the corpus (49 MB
    # for a 195k-memory store, 30+ s render time before Chrome gave up).
    return {
        "nodes": nodes[:NODE_LIMIT], "edges": edges,
        "categories": categories, "weights": weights,
        "dream_sessions": dream_sessions,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": actual_dim, "source": "SQLite",
            "path": db_path, "query_ms": ms,
        },
    }


def read_postgres() -> dict:
    """Read live graph data from the pgvector pod (Pro/Enterprise tier).

    Connection params come from MM_POSTGRES_* env (host/port/db/user/
    password) or MM_POSTGRES_DSN.  Schema mirrors SQLiteStore: memories,
    connections, dream_sessions.  Same NODE_LIMIT / EDGE_LIMIT cap as
    the SQLite path so the bootstrap HTML stays sane on a 245k-memory,
    784k-edge corpus.
    """
    import psycopg
    t0 = time.perf_counter()
    dsn = os.environ.get("MM_POSTGRES_DSN") or (
        f"host={os.environ.get('MM_POSTGRES_HOST', '127.0.0.1')} "
        f"port={os.environ.get('MM_POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('MM_POSTGRES_DB', 'mazemaker')} "
        f"user={os.environ.get('MM_POSTGRES_USER', 'mazemaker')} "
        f"password={os.environ.get('MM_POSTGRES_PASSWORD', '')}"
    )
    with psycopg.connect(dsn, connect_timeout=8) as conn:
        return _read_postgres_body(conn, t0)


def _read_postgres_body(conn, t0):
    cur = conn.cursor()

    # Top-N hubs ordered by total degree.  Postgres has window-friendly
    # indexes on connections.source_id / target_id, so a single query
    # with two LATERAL joins beats the two-subquery shape we use on
    # SQLite (where window functions are slower without statistics).
    cur.execute(f"""
        SELECT m.id, m.label, length(m.content) AS content_length,
               m.salience, m.access_count, m.created_at,
               COALESCE(out_d.out_degree, 0) AS out_degree,
               COALESCE(in_d.in_degree, 0)  AS in_degree,
               COALESCE(out_d.avg_weight, 0) AS avg_weight
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree, AVG(weight) AS avg_weight
                   FROM connections GROUP BY source_id) out_d ON m.id = out_d.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) in_d ON m.id = in_d.target_id
        ORDER BY (COALESCE(out_d.out_degree, 0) + COALESCE(in_d.in_degree, 0)) DESC
        LIMIT %s
    """, (NODE_LIMIT,))
    import datetime as _dt
    def _to_epoch(v):
        if v is None: return 0.0
        if isinstance(v, _dt.datetime): return v.timestamp()
        try: return float(v)
        except (TypeError, ValueError): return 0.0

    nodes = []
    for r in cur.fetchall():
        lbl = r[1] or ""
        nodes.append({
            "id": r[0], "label": lbl[:50], "category": _categorize(lbl),
            "content_length": r[2] or 0,
            "salience": float(r[3] or 1.0),
            "access_count": r[4] or 0,
            "created_at": _to_epoch(r[5]),
            "out_degree": r[6], "in_degree": r[7],
            "total_degree": r[6] + r[7],
            "avg_weight": round(float(r[8] or 0), 4),
        })

    hub_ids = [n["id"] for n in nodes]
    if hub_ids:
        cur.execute(
            "SELECT source_id, target_id, weight FROM connections "
            "WHERE source_id = ANY(%s) AND target_id = ANY(%s) "
            "ORDER BY weight DESC LIMIT %s",
            (hub_ids, hub_ids, EDGE_LIMIT),
        )
        edges = [{"source": r[0], "target": r[1],
                  "weight": round(float(r[2]), 4)} for r in cur.fetchall()]
    else:
        edges = []

    cur.execute("""
        SELECT cat, COUNT(*) FROM (
            SELECT CASE
                WHEN label LIKE 'peer:%%'                        THEN 'Peer'
                WHEN label LIKE 'turn:%%' OR label LIKE 'msg:%%' THEN 'Conversation'
                WHEN label LIKE 'session:%%'                     THEN 'Session'
                WHEN label LIKE 'doc:%%'                         THEN 'Document'
                WHEN label LIKE 'skill:%%'                       THEN 'Skill'
                ELSE 'Other'
            END AS cat FROM memories
        ) sub GROUP BY cat ORDER BY COUNT(*) DESC
    """)
    categories = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("""
        SELECT bucket, COUNT(*) FROM (
            SELECT CASE
                WHEN weight >= 0.8 THEN 'Strong (0.8-1.0)'
                WHEN weight >= 0.6 THEN 'Med-Strong (0.6-0.8)'
                WHEN weight >= 0.4 THEN 'Medium (0.4-0.6)'
                WHEN weight >= 0.2 THEN 'Weak (0.2-0.4)'
                ELSE 'Very Weak (0-0.2)'
            END AS bucket FROM connections
        ) sub GROUP BY bucket
    """)
    weights = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) FROM memories");      n_mem  = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections");   n_conn = cur.fetchone()[0]

    actual_dim = 1024
    try:
        cur.execute("SELECT vector_dims(embedding) FROM memories "
                    "WHERE embedding IS NOT NULL LIMIT 1")
        row = cur.fetchone()
        if row:
            actual_dim = int(row[0])
    except Exception:
        pass

    dream_sessions: list[dict] = []
    try:
        cur.execute(
            "SELECT id, phase, started_at, finished_at, "
            "       memories_processed, connections_strengthened, "
            "       connections_pruned, bridges_found, insights_created "
            "FROM dream_sessions ORDER BY started_at"
        )
        for r in cur.fetchall():
            stats_blob = json.dumps({
                "memories_processed": r[4],
                "connections_strengthened": r[5],
                "connections_pruned": r[6],
                "bridges_found": r[7],
                "insights_created": r[8],
            })
            dream_sessions.append({
                "id": r[0], "phase": r[1],
                "started_at": float(r[2]) if r[2] else 0,
                "completed_at": float(r[3]) if r[3] else None,
                "stats": stats_blob,
            })
    except Exception:
        pass

    ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["db_query_ms"] = ms
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "dream_sessions": dream_sessions,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": actual_dim, "source": "Postgres",
            "path": f"{os.environ.get('MM_POSTGRES_HOST', '127.0.0.1')}:"
                    f"{os.environ.get('MM_POSTGRES_PORT', '5432')}/"
                    f"{os.environ.get('MM_POSTGRES_DB', 'mazemaker')}",
            "query_ms": ms,
        },
    }


def read_mssql(dsn: str) -> dict:
    import pyodbc
    t0 = time.perf_counter()
    conn = pyodbc.connect(dsn, timeout=10)
    try:
        return _read_mssql_body(conn, t0, dsn)
    finally:
        conn.close()


def _read_mssql_body(conn, t0, dsn):
    """MSSQL-aware reader.

    The MSSQL store has TWO memory tables:
      * `NeuralMemory` (~696k rows) — bulk vector archive. surrogate_id pk,
         metadata_json carries the label, NO content text column, no graph edges.
      * `memories` (~25k rows) — legacy text-bearing slice whose ids are
         the endpoints of the `connections` graph (~8k edges).

    For the dashboard we want the BIG numbers from NeuralMemory (that's the
    real "Wissen") but the graph viz from `memories`+`connections` (that's
    where the edges live). We fold both into one payload so the user sees
    the full picture without losing the live graph.
    """
    cur = conn.cursor()

    # ── headline counts ──────────────────────────────────────────────────
    n_nm = _cache_get("nm_count", CACHE_TTL_NM_COUNT)
    if n_nm is None:
        cur.execute("SELECT COUNT(*) FROM NeuralMemory")
        n_nm = cur.fetchone()[0]
        _cache_set("nm_count", n_nm, CACHE_TTL_NM_COUNT)
    cur.execute("SELECT COUNT(*) FROM memories");            n_legacy = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections");         n_conn = cur.fetchone()[0]
    # NeuralMemory.legacy_id loosely subsumes the `memories` table, so we
    # report n_nm as the headline rather than paying for an exact dedupe
    # subquery on every poll. ~3% overlap with `memories` is rounding error
    # at this scale.
    n_total = n_nm

    actual_dim = _cache_get("nm_dim", CACHE_TTL_DIM)
    if actual_dim is None:
        # No ORDER BY — any row's vector_dim is fine; sorting on a 696k
        # table without a covering index costs ~2s.
        cur.execute("SELECT TOP 1 vector_dim FROM NeuralMemory")
        emb_row    = cur.fetchone()
        actual_dim = emb_row[0] if emb_row else 1024
        _cache_set("nm_dim", actual_dim, CACHE_TTL_DIM)

    # ── categories: union over NeuralMemory.metadata_json + memories.label ─
    # The NM scan is the slowest single query (~1.4s on 696k); cache 60s.
    categories = _cache_get("nm_categories", CACHE_TTL_CATS)
    if categories is None:
        cur.execute("""
            SELECT cat, COUNT(*) FROM (
                SELECT CASE
                    WHEN metadata_json LIKE '%"label":"peer:%'      THEN 'Peer'
                    WHEN metadata_json LIKE '%"label":"msg:%'       THEN 'Conversation'
                    WHEN metadata_json LIKE '%"label":"turn%'       THEN 'Conversation'
                    WHEN metadata_json LIKE '%"label":"archive:%'   THEN 'Archive'
                    WHEN metadata_json LIKE '%"label":"session:%'   THEN 'Session'
                    WHEN metadata_json LIKE '%"label":"derived:%'   THEN 'Derived'
                    WHEN metadata_json LIKE '%"label":"doc:%'       THEN 'Document'
                    WHEN metadata_json LIKE '%"label":"skill:%'     THEN 'Skill'
                    WHEN metadata_json LIKE '%"label":"test_%'      THEN 'Test/Bench'
                    WHEN metadata_json LIKE '%"label":"bench-%'     THEN 'Test/Bench'
                    WHEN metadata_json LIKE '%"label":"commit:%'    THEN 'Commit'
                    WHEN metadata_json LIKE '%"label":"bug:%'       THEN 'Bug'
                    WHEN metadata_json LIKE '%"label":"decision:%'  THEN 'Decision'
                    WHEN metadata_json LIKE '%"label":"ops:%'       THEN 'Ops'
                    WHEN metadata_json LIKE '%"label":"signal:%'    THEN 'Signal'
                    WHEN metadata_json LIKE '%"label":"invariant:%' THEN 'Invariant'
                    WHEN metadata_json LIKE '%"label":"fact:%'      THEN 'Fact'
                    ELSE 'Other'
                END AS cat FROM NeuralMemory
            ) AS x GROUP BY cat ORDER BY COUNT(*) DESC
        """)
        cats_nm = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("""
            SELECT cat, COUNT(*) FROM (
                SELECT CASE
                    WHEN label LIKE 'peer:%'                              THEN 'Peer'
                    WHEN label LIKE 'turn%' OR label LIKE 'msg:%'         THEN 'Conversation'
                    WHEN label LIKE 'archive:%'                           THEN 'Archive'
                    WHEN label LIKE 'session-summary' OR label LIKE 'session:%' THEN 'Session'
                    WHEN label LIKE 'derived:%'                           THEN 'Derived'
                    WHEN label LIKE 'doc:%'                               THEN 'Document'
                    WHEN label LIKE 'skill:%'                             THEN 'Skill'
                    WHEN label LIKE 'test_%' OR label LIKE 'bench-%'      THEN 'Test/Bench'
                    WHEN label LIKE 'commit:%'                            THEN 'Commit'
                    WHEN label LIKE 'bug:%'                               THEN 'Bug'
                    WHEN label LIKE 'decision:%'                          THEN 'Decision'
                    WHEN label LIKE 'ops:%'                               THEN 'Ops'
                    WHEN label LIKE 'signal:%'                            THEN 'Signal'
                    WHEN label LIKE 'invariant:%'                         THEN 'Invariant'
                    WHEN label LIKE 'fact:%'                              THEN 'Fact'
                    WHEN label LIKE 'pre-compress'                        THEN 'Session'
                    WHEN label LIKE '%-msg' OR label LIKE 'asst-msg' OR label LIKE 'user-msg' THEN 'Conversation'
                    ELSE 'Other'
                END AS cat FROM memories
            ) AS x GROUP BY cat
        """)
        cats_legacy = {r[0]: r[1] for r in cur.fetchall()}

        merged = dict(cats_nm)
        for k, v in cats_legacy.items():
            merged[k] = merged.get(k, 0) + v
        categories = [{"name": k, "count": v} for k, v in
                      sorted(merged.items(), key=lambda kv: -kv[1])]
        _cache_set("nm_categories", categories, CACHE_TTL_CATS)

    # ── connection-weight distribution (only graph edges that exist) ─────
    cur.execute("""
        SELECT bucket, COUNT(*) FROM (
            SELECT CASE
                WHEN weight >= 0.8 THEN 'Strong (0.8-1.0)'
                WHEN weight >= 0.6 THEN 'Med-Strong (0.6-0.8)'
                WHEN weight >= 0.4 THEN 'Medium (0.4-0.6)'
                WHEN weight >= 0.2 THEN 'Weak (0.2-0.4)'
                ELSE 'Very Weak (0-0.2)'
            END AS bucket FROM connections
        ) AS x GROUP BY bucket
    """)
    weights = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    # ── graph viz: top hub nodes from `memories` (only place with edges) ─
    cur.execute(f"""
        SELECT TOP {NODE_LIMIT * 5}
               m.id, m.label, m.content, m.salience, m.access_count, m.created_at,
               COALESCE(out_d.out_degree, 0),
               COALESCE(in_d.in_degree,  0),
               COALESCE(out_d.avg_weight, 0)
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree, AVG(weight) AS avg_weight
                   FROM connections GROUP BY source_id) out_d ON m.id = out_d.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) in_d ON m.id = in_d.target_id
        ORDER BY (COALESCE(out_d.out_degree,0) + COALESCE(in_d.in_degree,0)) DESC
    """)
    nodes = []
    for r in cur.fetchall():
        lbl = r[1] or ""
        created_at = r[5].timestamp() if r[5] else 0
        nodes.append({
            "id": r[0], "label": lbl[:50], "category": _categorize(lbl),
            "content_length": len(r[2]) if r[2] else 0,
            "salience": r[3] or 1.0, "access_count": r[4] or 0,
            "created_at": created_at,
            "out_degree": r[6], "in_degree": r[7], "total_degree": r[6] + r[7],
            "avg_weight": round(r[8], 4),
        })

    hub_ids = [n["id"] for n in nodes[:NODE_LIMIT]]
    if hub_ids:
        placeholders = ",".join("?" * len(hub_ids))
        cur.execute(
            f"SELECT TOP {int(EDGE_LIMIT)} source_id, target_id, weight FROM connections "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) "
            f"ORDER BY weight DESC",
            hub_ids + hub_ids,
        )
        edges = [{"source": r[0], "target": r[1], "weight": round(r[2], 4)} for r in cur.fetchall()]
    else:
        edges = []

    # ── dream sessions (MSSQL schema differs from SQLite) ────────────────
    dream_sessions: list[dict] = []
    try:
        cur.execute("""
            SELECT id, phase, started_at, finished_at,
                   memories_processed, connections_strengthened, connections_pruned,
                   bridges_found, insights_created
            FROM dream_sessions ORDER BY started_at
        """)
        for r in cur.fetchall():
            stats_blob = json.dumps({
                "memories_processed":       r[4],
                "connections_strengthened": r[5],
                "connections_pruned":       r[6],
                "bridges_found":            r[7],
                "insights_created":         r[8],
            })
            dream_sessions.append({
                "id": r[0], "phase": r[1],
                "started_at":   r[2].timestamp() if r[2] else 0,
                "completed_at": r[3].timestamp() if r[3] else 0,
                "stats":        stats_blob,
            })
    except Exception as exc:
        logger.debug(f"dream_sessions skipped: {exc}")

    # ── extras (footer/sidebar context) ──────────────────────────────────
    extras = {
        "neuralmemory_rows": n_nm,
        "legacy_memories_rows": n_legacy,
    }
    try:
        cur.execute("SELECT COUNT(*) FROM connection_history")
        extras["connection_history_rows"] = cur.fetchone()[0]
    except Exception:
        pass
    try:
        cur.execute("SELECT COUNT(*) FROM dream_insights")
        extras["dream_insights_rows"] = cur.fetchone()[0]
    except Exception:
        pass

    ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["db_query_ms"] = ms
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "dream_sessions": dream_sessions,
        "stats": {
            "memories": n_total, "connections": n_conn,
            "embedding_dim": actual_dim, "source": "MSSQL",
            "path": "mssql://NeuralMemory@127.0.0.1:1433", "query_ms": ms,
            **extras,
        },
    }


def _build_mssql_dsn(args) -> str | None:
    """Resolve MSSQL DSN from --mssql-dsn or the hermes config backup."""
    if args.mssql_dsn:
        return args.mssql_dsn
    # Fall back to the pre-flip backup, since the live config got scrubbed.
    backup = os.path.expanduser("~/.hermes/config.yaml.backup-pre-mcp-flip-20260501-005028")
    if not os.path.exists(backup):
        return None
    try:
        import yaml
        with open(backup) as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("memory", {}).get("neural", {}).get("dream", {}).get("mssql")
        if not m:
            return None
        return (
            f"DRIVER={m['driver']};SERVER={m['server']},1433;DATABASE={m['database']};"
            f"UID={m['username']};PWD={m['password']};"
            "TrustServerCertificate=yes;Encrypt=optional"
        )
    except Exception as exc:
        logger.warning(f"could not parse mssql config from backup: {exc}")
        return None


_OFFLINE_SNAPSHOT = {
    "nodes": [], "edges": [],
    "categories": [], "weights": [],
    "dream_sessions": [],
    "stats": {
        "memories": 0, "connections": 0,
        "embedding_dim": 0, "source": "offline",
        "path": "", "query_ms": 0,
    },
}


def _safe_read(read_fn, label: str) -> dict:
    """Wrap a backend read so DB outages degrade to the offline snapshot
    rather than crash the service. The poll loop will keep retrying."""
    try:
        return read_fn()
    except Exception as exc:
        logger.warning(f"{label} unreachable, serving offline snapshot: {exc}")
        snap = dict(_OFFLINE_SNAPSHOT)
        snap["stats"] = dict(_OFFLINE_SNAPSHOT["stats"])
        snap["stats"]["source"] = f"{label} (offline)"
        snap["stats"]["error"]  = str(exc)[:200]
        return snap


def load_data(args) -> tuple[dict, callable]:
    # Auto-promote to Postgres if MM_DB_BACKEND=postgres in env (matches
    # the engine's own dispatch) — operator dashboards on Pro/Enterprise
    # tier should mirror what the live engine reads from.
    if args.source == "sqlite" and (
        os.environ.get("MM_DB_BACKEND", "").lower() == "postgres"
    ):
        args.source = "postgres"

    if args.source == "postgres":
        read_fn = lambda: read_postgres()
        data = _safe_read(read_fn, "Postgres")
        s = data["stats"]
        logger.info(
            f"Postgres: {s['memories']}M  {s['connections']}C  "
            f"({s['query_ms']}ms)  @{s.get('path', '')}"
        )
        _metrics["db_source"] = "Postgres"
        return data, lambda: _safe_read(read_fn, "Postgres")

    if args.source == "mssql":
        dsn = _build_mssql_dsn(args)
        if not dsn:
            logger.error("--source=mssql but no DSN found (tried --mssql-dsn + hermes backup)")
            sys.exit(1)
        read_fn = lambda: read_mssql(dsn)
        data = _safe_read(read_fn, "MSSQL")
        s = data["stats"]
        extras_str = " ".join(f"{k}={v}" for k, v in s.items() if k.endswith("_rows"))
        logger.info(f"MSSQL: {s['memories']}M  {s['connections']}C  ({s['query_ms']}ms)  {extras_str}")
        _metrics["db_source"] = "MSSQL"
        return data, lambda: _safe_read(read_fn, "MSSQL")

    db_path = args.db or DEFAULT_SQLITE
    read_fn = lambda: read_sqlite(db_path)
    if not os.path.exists(db_path):
        logger.warning(f"Database not found: {db_path} — serving offline snapshot")
        _metrics["db_source"] = "SQLite"
        return _safe_read(lambda: (_ for _ in ()).throw(FileNotFoundError(db_path)), "SQLite"), \
               lambda: _safe_read(read_fn, "SQLite")
    data = _safe_read(read_fn, "SQLite")
    s = data["stats"]
    logger.info(f"SQLite: {s['memories']}M  {s['connections']}C  ({s['query_ms']}ms)")
    _metrics["db_source"] = "SQLite"
    return data, lambda: _safe_read(read_fn, "SQLite")


# ═══════════════════════════════════════════════════════════
# WebSocket connection manager
# ═══════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self._conns: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._conns.append(ws)
        _metrics["ws_total"] += 1

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._conns:
                self._conns.remove(ws)

    async def broadcast(self, message: dict):
        t0 = time.perf_counter()
        async with self._lock:
            conns = list(self._conns)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._conns:
                        self._conns.remove(ws)
        _metrics["broadcast_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    @property
    def count(self) -> int:
        return len(self._conns)


manager       = ConnectionManager()
current_data: dict     = {}
current_hash: str      = ""
reload_fn:    callable = None
_poll_interval: float  = POLL_INTERVAL

# Architect-mirror state: highest IDs seen on the previous poll, used
# to compute incremental events.  Reset on db source switch.
_mirror_state: dict = {
    "max_memory_id": 0,
    "max_edge_changed_at": 0.0,   # connection_history.changed_at high-water
    "dream_session_states": {},   # session_id → {phase, started_at, finished_at}
    "last_event_seq": 0,           # rolling sequence for clients
}


def compute_data_hash(data: dict) -> str:
    s = data["stats"]
    return f"{s['memories']}:{s['connections']}"


# ═══════════════════════════════════════════════════════════
# Architect mirror — incremental event reader
# ═══════════════════════════════════════════════════════════

def read_mirror_events_postgres() -> list[dict]:
    """Pull events from Postgres that happened since the last poll.

    Three event types:
    * memory.added  — new rows in `memories` (id > previous high-water)
    * edge.changed  — new rows in `connection_history`
                      (changed_at > previous high-water).  This is the
                      table the dream engine writes to on every NREM
                      strengthen / REM bridge / Insight link, plus the
                      regular auto-connect path on memory writes.
    * dream.phase.{started,finished} — transitions on `dream_sessions`
                      (each session's started_at / finished_at).

    Each event carries `agent` + `color` derived from label patterns,
    so the mirror frontend can render without a per-row lookup.
    """
    import psycopg
    import datetime as _dt

    def _to_epoch(v):
        if v is None: return 0.0
        if isinstance(v, _dt.datetime): return v.timestamp()
        try: return float(v)
        except (TypeError, ValueError): return 0.0

    dsn = os.environ.get("MM_POSTGRES_DSN") or (
        f"host={os.environ.get('MM_POSTGRES_HOST', '127.0.0.1')} "
        f"port={os.environ.get('MM_POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('MM_POSTGRES_DB', 'mazemaker')} "
        f"user={os.environ.get('MM_POSTGRES_USER', 'mazemaker')} "
        f"password={os.environ.get('MM_POSTGRES_PASSWORD', '')}"
    )

    events: list[dict] = []
    seq = _mirror_state["last_event_seq"]
    now = time.time()

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        cur = conn.cursor()

        # 1. New memories
        prev_id = _mirror_state["max_memory_id"]
        cur.execute(
            "SELECT id, label, created_at FROM memories "
            "WHERE id > %s ORDER BY id ASC LIMIT 500",
            (prev_id,),
        )
        new_max_id = prev_id
        for r in cur.fetchall():
            mid, label, created = r[0], r[1] or "", r[2]
            agent = _classify_agent(label)
            seq += 1
            events.append({
                "seq": seq, "t": "memory.added",
                "id": mid, "label": label[:80],
                "agent": agent, "color": _agent_color(agent),
                "ts": _to_epoch(created), "wall": now,
            })
            if mid > new_max_id: new_max_id = mid
        _mirror_state["max_memory_id"] = new_max_id

        # 2. Edge events from connection_history
        prev_ts = _mirror_state["max_edge_changed_at"]
        # connection_history is the dream engine's audit log:
        # source_id, target_id, old_weight, new_weight, reason, changed_at, dream_session_id
        try:
            cur.execute(
                "SELECT source_id, target_id, old_weight, new_weight, "
                "       reason, changed_at, dream_session_id "
                "FROM connection_history "
                "WHERE changed_at > %s ORDER BY changed_at ASC LIMIT 1000",
                (prev_ts,),
            )
            new_max_ts = prev_ts
            for r in cur.fetchall():
                src, dst, oldw, neww, reason, changed_at, dsid = r
                ts = _to_epoch(changed_at)
                seq += 1
                events.append({
                    "seq": seq, "t": "edge.changed",
                    "src": src, "dst": dst,
                    "old_weight": float(oldw or 0),
                    "new_weight": float(neww or 0),
                    "reason": reason or "auto",
                    "dream_session_id": dsid,
                    "ts": ts, "wall": now,
                })
                if ts > new_max_ts: new_max_ts = ts
            _mirror_state["max_edge_changed_at"] = new_max_ts
        except psycopg.errors.UndefinedTable:
            # connection_history table may not exist on older deployments
            pass

        # 3. Dream session phase transitions
        try:
            cur.execute(
                "SELECT id, phase, started_at, finished_at, "
                "       memories_processed, connections_strengthened, "
                "       connections_pruned, bridges_found, insights_created "
                "FROM dream_sessions "
                "WHERE started_at >= %s ORDER BY started_at ASC",
                (now - 600,),  # last 10 minutes window
            )
            states = _mirror_state["dream_session_states"]
            for r in cur.fetchall():
                sid, phase, started, finished = r[0], r[1], r[2], r[3]
                started_ts = _to_epoch(started)
                finished_ts = _to_epoch(finished) if finished else 0.0
                stats_blob = {
                    "memories_processed":      r[4] or 0,
                    "connections_strengthened": r[5] or 0,
                    "connections_pruned":      r[6] or 0,
                    "bridges_found":           r[7] or 0,
                    "insights_created":        r[8] or 0,
                }
                prev = states.get(sid, {})
                if not prev:
                    seq += 1
                    events.append({
                        "seq": seq, "t": "dream.phase.started",
                        "session_id": sid, "phase": (phase or "").upper(),
                        "ts": started_ts, "wall": now,
                    })
                if finished_ts and not prev.get("finished_at"):
                    seq += 1
                    events.append({
                        "seq": seq, "t": "dream.phase.finished",
                        "session_id": sid, "phase": (phase or "").upper(),
                        "stats": stats_blob,
                        "ts": finished_ts, "wall": now,
                    })
                states[sid] = {
                    "phase": phase, "started_at": started_ts,
                    "finished_at": finished_ts, "stats": stats_blob,
                }
            # Garbage-collect old session entries
            old_cutoff = now - 3600
            states_keep = {
                k: v for k, v in states.items()
                if v.get("started_at", 0) > old_cutoff
            }
            _mirror_state["dream_session_states"] = states_keep
        except psycopg.errors.UndefinedTable:
            pass

    _mirror_state["last_event_seq"] = seq
    return events


def reset_mirror_state() -> None:
    """Re-baseline mirror state to *now* so we don't replay pre-startup
    history on the first poll.  Called once at startup with the current
    DB high-water marks."""
    import psycopg, datetime as _dt
    def _to_epoch(v):
        if v is None: return 0.0
        if isinstance(v, _dt.datetime): return v.timestamp()
        try: return float(v)
        except (TypeError, ValueError): return 0.0
    if os.environ.get("MM_DB_BACKEND", "").lower() != "postgres":
        return
    dsn = os.environ.get("MM_POSTGRES_DSN") or (
        f"host={os.environ.get('MM_POSTGRES_HOST', '127.0.0.1')} "
        f"port={os.environ.get('MM_POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('MM_POSTGRES_DB', 'mazemaker')} "
        f"user={os.environ.get('MM_POSTGRES_USER', 'mazemaker')} "
        f"password={os.environ.get('MM_POSTGRES_PASSWORD', '')}"
    )
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM memories")
            _mirror_state["max_memory_id"] = int(cur.fetchone()[0])
            try:
                cur.execute("SELECT COALESCE(MAX(changed_at), 0) "
                            "FROM connection_history")
                row = cur.fetchone()
                _mirror_state["max_edge_changed_at"] = _to_epoch(row[0]) if row else 0.0
            except Exception:
                _mirror_state["max_edge_changed_at"] = 0.0
        logger.info(
            "mirror baseline: max_memory_id=%d max_edge_ts=%.0f",
            _mirror_state["max_memory_id"],
            _mirror_state["max_edge_changed_at"],
        )
    except Exception as exc:
        logger.warning("mirror baseline failed: %s", exc)


async def poll_db_changes():
    global current_data, current_hash
    loop = asyncio.get_event_loop()
    while True:
        try:
            _metrics["poll_count"] += 1
            new_data = await loop.run_in_executor(_DB_EXECUTOR, reload_fn)
            new_hash = compute_data_hash(new_data)

            # Mirror events run on every poll, independent of the
            # graph-snapshot diff: e.g. dream phases tick even if the
            # node count hasn't moved.
            mirror_events: list[dict] = []
            if os.environ.get("MM_DB_BACKEND", "").lower() == "postgres":
                try:
                    mirror_events = await loop.run_in_executor(
                        _DB_EXECUTOR, read_mirror_events_postgres
                    )
                except Exception as exc:
                    logger.debug("mirror events poll error: %s", exc)
                if mirror_events and manager.count:
                    await manager.broadcast({
                        "type": "mirror_events",
                        "events": mirror_events,
                    })

            if new_hash != current_hash:
                old_m = current_data.get("stats", {}).get("memories",    0)
                old_c = current_data.get("stats", {}).get("connections",  0)
                delta = {
                    "type":   "update",
                    "stats":  new_data["stats"],
                    "delta":  {
                        "new_memories":    new_data["stats"]["memories"]    - old_m,
                        "new_connections": new_data["stats"]["connections"] - old_c,
                    },
                    "nodes":      new_data["nodes"],
                    "edges":      new_data["edges"],
                    "categories": new_data["categories"],
                    "weights":    new_data["weights"],
                }
                current_data = new_data
                current_hash = new_hash
                _metrics["last_update_time"] = time.time()
                if manager.count:
                    await manager.broadcast(delta)
                    _metrics["updates_sent"] += 1
                    logger.info(
                        f"↑ update  {current_data['stats']['memories']}M  "
                        f"{current_data['stats']['connections']}C  "
                        f"db={_metrics['db_query_ms']:.0f}ms  "
                        f"bcast={_metrics['broadcast_ms']:.1f}ms  "
                        f"clients={manager.count}"
                    )
        except Exception as exc:
            logger.error(f"DB poll error: {exc}")
        await asyncio.sleep(_poll_interval)


# ═══════════════════════════════════════════════════════════
# PTY terminal session
# ═══════════════════════════════════════════════════════════

class TerminalSession:
    """One PTY-backed interactive shell, bridged to a WebSocket."""

    def __init__(self, ws: WebSocket):
        self.ws         = ws
        self.master_fd  = -1
        self.proc       = None

    async def run(self):
        loop = asyncio.get_event_loop()
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd

        shell = os.environ.get("SHELL", "/usr/bin/bash")
        env   = {
            **os.environ,
            "TERM":           "xterm-256color",
            "COLORTERM":      "truecolor",
            "NEURAL_DASHBOARD": "1",
        }
        self.proc = await asyncio.create_subprocess_exec(
            shell,
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, preexec_fn=os.setsid, close_fds=True,
        )
        os.close(slave_fd)

        pty_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)

        def _on_readable():
            try:
                data = os.read(master_fd, 8192)
                loop.call_soon_threadsafe(pty_q.put_nowait, data)
            except OSError:
                loop.call_soon_threadsafe(pty_q.put_nowait, None)

        loop.add_reader(master_fd, _on_readable)
        try:
            await asyncio.gather(
                self._pty_to_ws(pty_q),
                self._ws_to_pty(),
                return_exceptions=True,
            )
        finally:
            loop.remove_reader(master_fd)
            await self._teardown()

    async def _pty_to_ws(self, queue: asyncio.Queue):
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            try:
                await self.ws.send_bytes(chunk)
            except Exception:
                return

    async def _ws_to_pty(self):
        while True:
            try:
                msg = await self.ws.receive()
            except WebSocketDisconnect:
                return
            mtype = msg.get("type", "")
            if mtype == "websocket.disconnect":
                return
            if mtype != "websocket.receive":
                continue
            raw_bytes = msg.get("bytes")
            raw_text  = msg.get("text")
            if raw_bytes:
                try:
                    os.write(self.master_fd, raw_bytes)
                except OSError:
                    return
            elif raw_text:
                try:
                    obj = json.loads(raw_text)
                    if obj.get("type") == "resize":
                        rows = max(1, int(obj.get("rows", 24)))
                        cols = max(1, int(obj.get("cols", 80)))
                        winsize = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
                except Exception:
                    pass

    async def _teardown(self):
        if self.master_fd >= 0:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = -1
        if self.proc:
            try:
                self.proc.kill()
                await asyncio.wait_for(self.proc.wait(), timeout=2.0)
            except Exception:
                pass
            self.proc = None


# ═══════════════════════════════════════════════════════════
# HTML builder — inlines all cached JS/CSS libs
# ═══════════════════════════════════════════════════════════

def _build_html(template_name: str) -> str:
    path = TEMPLATE_DIR / template_name
    if not path.exists():
        path = TEMPLATE_DIR / "template.html"
    html     = path.read_text()
    libs_dir = TEMPLATE_DIR / ".lib_cache"

    js_subs = {
        "__PLOTLY_SCRIPT__":       "plotly-2.27.0.min.js",
        "__THREEJS_SCRIPT__":      "three-0.160.0.min.js",
        "__FORCEGRAPH_SCRIPT__":   "3d-force-graph.min.js",
        "__XTERM_SCRIPT__":        "xterm-4.19.0.min.js",
        "__XTERM_FIT_SCRIPT__":    "xterm-addon-fit-0.5.0.min.js",
    }
    for placeholder, filename in js_subs.items():
        f = libs_dir / filename
        html = html.replace(placeholder, f"<script>{f.read_text()}</script>" if f.exists() else "")

    css_file = libs_dir / "xterm-4.19.0.css"
    html = html.replace(
        "__XTERM_CSS__",
        f"<style>{css_file.read_text()}</style>" if css_file.exists() else "",
    )

    html = html.replace("__DATA_JSON__", json.dumps(current_data))
    return html


# ═══════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_data, current_hash, reload_fn
    # Baseline mirror state once before the poll loop starts so we
    # don't dump pre-startup history into the first broadcast.
    try:
        reset_mirror_state()
    except Exception as exc:
        logger.debug("mirror baseline skipped: %s", exc)
    task = asyncio.create_task(poll_db_changes())
    yield
    task.cancel()
    _DB_EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Mazemaker Live Dashboard", lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(_build_html("template-live.html"))


@app.get("/desktop")
async def desktop():
    return HTMLResponse(_build_html("template-desktop.html"))


@app.get("/mirror")
async def mirror():
    """The Architect Mirror — live reflection of every memory write,
    edge change, and dream-phase transition across all MCP clients.
    Subscribes to the same /ws/stream as the main dashboard but
    consumes the `mirror_events` payload."""
    return HTMLResponse(_build_html("template-mirror.html"))


@app.get("/api/mirror/agents")
async def api_mirror_agents():
    """Operator-side palette + label-prefix list so the mirror frontend
    can render its legend without hardcoding the classifier."""
    return {"palette": AGENT_PALETTE}


@app.get("/api/graph")
async def api_graph():
    return current_data


@app.get("/api/stats")
async def api_stats():
    return current_data.get("stats", {})


@app.get("/api/metrics")
async def api_metrics():
    uptime = time.time() - _metrics["start_time"]
    return {
        "uptime_s":         round(uptime, 1),
        "db_query_ms":      _metrics["db_query_ms"],
        "broadcast_ms":     _metrics["broadcast_ms"],
        "updates_sent":     _metrics["updates_sent"],
        "ws_clients":       manager.count,
        "ws_total":         _metrics["ws_total"],
        "last_update_time": _metrics["last_update_time"],
        "poll_count":       _metrics["poll_count"],
        "poll_interval_s":  _poll_interval,
        "db_source":        _metrics["db_source"],
    }


@app.get("/api/health")
async def api_health():
    s = current_data.get("stats", {})
    return {
        "status":      "ok",
        "uptime_s":    round(time.time() - _metrics["start_time"], 1),
        "db":          _metrics["db_source"],
        "memories":    s.get("memories",    0),
        "connections": s.get("connections", 0),
        "ws_clients":  manager.count,
    }


@app.post("/api/restart")
async def api_restart():
    import subprocess
    try:
        subprocess.Popen(
            ["systemctl", "--user", "restart", "neural-dashboard.service"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return {"status": "restarting", "message": "Dashboard service restarting…"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type":       "initial",
            "stats":      current_data.get("stats", {}),
            "nodes":      current_data.get("nodes", []),
            "edges":      current_data.get("edges", []),
            "categories": current_data.get("categories", []),
            "weights":    current_data.get("weights", []),
        })
        while True:
            raw = await ws.receive_text()
            if raw == "ping":
                await ws.send_json({"type": "pong", "t": time.time()})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await manager.disconnect(ws)


@app.websocket("/ws/terminal")
async def websocket_terminal(ws: WebSocket):
    await ws.accept()
    session = TerminalSession(ws)
    try:
        await session.run()
    except Exception as exc:
        logger.debug(f"Terminal session ended: {exc}")


# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════

def main():
    global current_data, current_hash, reload_fn, _poll_interval

    parser = argparse.ArgumentParser(description="Mazemaker Live Dashboard Server")
    parser.add_argument("--db",             default=None,  help="Force SQLite path")
    parser.add_argument("--source",         default="sqlite",
                        choices=("sqlite", "postgres", "mssql"),
                        help="Which backend to read from")
    parser.add_argument("--mssql-dsn",      default=None,
                        help="Full pyodbc connection string; falls back to hermes config backup")
    parser.add_argument("--port",           type=int,      default=8443)
    parser.add_argument("--host",           default="0.0.0.0")
    parser.add_argument("--no-tls",         action="store_true")
    parser.add_argument("--desktop-layer",  action="store_true")
    parser.add_argument("--watch-interval", type=float,    default=POLL_INTERVAL,
                        help="DB poll interval in seconds (default: 2.0)")
    args = parser.parse_args()

    if args.desktop_layer:
        logging.getLogger().setLevel(logging.WARNING)

    _poll_interval = args.watch_interval

    _ensure_libs()

    current_data, reload_fn = load_data(args)
    current_hash            = compute_data_hash(current_data)
    _metrics["db_source"]   = current_data["stats"]["source"]

    cert_dir  = TEMPLATE_DIR / ".certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = str(cert_dir / "dashboard.crt")
    key_file  = str(cert_dir / "dashboard.key")

    ssl_kwargs = {}
    if not args.no_tls:
        import subprocess as _sub
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            _sub.run([
                "openssl", "req", "-x509", "-newkey", "ec",
                "-pkeyopt", "ec_paramgen_curve:prime256v1",
                "-keyout", key_file, "-out", cert_file,
                "-days", "3650", "-nodes",
                "-subj", "/CN=mazemaker-dashboard/O=Mazemaker/C=DE",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0",
            ], capture_output=True, check=True)
        ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}

    proto = "https" if not args.no_tls else "http"
    ws_proto = proto.replace("http", "ws")
    s = current_data["stats"]

    print(f"\n{'═' * 64}")
    print(f"  ◈  Mazemaker — LIVE Dashboard  [{s['source']}]")
    print(f"  ◈  Dashboard : {proto}://localhost:{args.port}/")
    print(f"  ◈  Terminal  : {ws_proto}://localhost:{args.port}/ws/terminal")
    print(f"  ◈  Metrics   : {proto}://localhost:{args.port}/api/metrics")
    print(f"  ◈  Health    : {proto}://localhost:{args.port}/api/health")
    print(f"  ◈  {s['memories']} memories   {s['connections']} connections   dim={s['embedding_dim']}")
    print(f"  ◈  Poll every {_poll_interval}s   →   {args.host}:{args.port}")
    print(f"{'═' * 64}\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if not args.desktop_layer else "warning",
        access_log=not args.desktop_layer,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
