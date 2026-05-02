"""Dream Engine — autonomous background memory consolidation.

Implements three phases inspired by biological sleep:
  1. NREM — Replay & consolidation (strengthen, prune)
  2. REM  — Exploration & bridge discovery
  3. Insight — Abstraction & community detection

Runs as a background daemon during idle periods. Stores results
in the same SQLite DB as the main neural memory, extended with
dream-specific tables.

MSSQL support: if mssql_store is configured, dreams run against
the shared MSSQL backend (sneaky multi-agent consolidation).
Otherwise falls back to SQLite.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extensions for dream tables (SQLite)
# ---------------------------------------------------------------------------

_DREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    phase TEXT NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    bridges_found INTEGER DEFAULT 0,
    insights_created INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    insight_type TEXT NOT NULL,
    source_memory_id INTEGER,
    content TEXT,
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes
    ON connection_history(source_id, target_id);
"""


# ---------------------------------------------------------------------------
# Abstract Dream Backend
# ---------------------------------------------------------------------------

class DreamBackend:
    """Interface for dream storage backends (SQLite or MSSQL)."""

    def start_session(self, phase: str) -> int:
        raise NotImplementedError

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_connections(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def iter_connections(self) -> Iterator[Dict[str, Any]]:
        # Default: materialize via get_connections; subclasses can stream.
        return iter(self.get_connections())

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        raise NotImplementedError

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        raise NotImplementedError

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        raise NotImplementedError

    def prune_weak(self, threshold: float = 0.05) -> int:
        raise NotImplementedError

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        raise NotImplementedError

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite Dream Backend
# ---------------------------------------------------------------------------

class SQLiteDreamBackend(DreamBackend):
    """Dream backend using the existing neural memory SQLite DB."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_DREAM_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def start_session(self, phase: str) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
                (time.time(), phase)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dream_sessions SET "
                "finished_at = ?, memories_processed = ?, "
                "connections_strengthened = ?, connections_pruned = ?, "
                "bridges_found = ?, insights_created = ? WHERE id = ?",
                (
                    time.time(),
                    stats.get("processed", stats.get("explored", 0)),
                    stats.get("strengthened", 0),
                    stats.get("pruned", 0),
                    stats.get("bridges", 0),
                    stats.get("insights", 0),
                    session_id,
                )
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r["id"], "content": r["content"] or ""} for r in rows]
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT m.id, m.content,
                       (SELECT COUNT(*) FROM connections
                        WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at DESC LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": r["id"], "content": r["content"] or "", "connection_count": r["cnt"]}
                for r in rows
            ]
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        # Materializing helper for legacy callers; new code should prefer
        # iter_connections() to avoid loading the full graph into memory.
        return list(self.iter_connections())

    def iter_connections(self) -> Iterator[Dict[str, Any]]:
        """Stream live connections in 50K-row chunks, filtered by valid_to.

        D6 second-order root cause (Hermes 2026-05-02): the previous
        fetchall() pulled all 7.7M rows into a Python heap before NREM
        ran a single comparison. Streaming bounds resident memory and lets
        NREM start processing immediately. The valid_to IS NULL filter
        excludes temporal tombstones from the NREM sweep — pruned edges
        should not be re-weakened on every cycle.

        Defensive fallback: pre-bi-temporal DBs lack valid_to. Query
        without the filter on those (per-commit-reviewer 8b26966).
        """
        conn = self._connect()
        try:
            has_valid_to = any(
                r[1] == "valid_to"
                for r in conn.execute("PRAGMA table_info(connections)")
            )
            if has_valid_to:
                sql = ("SELECT source_id, target_id, weight FROM connections "
                       "WHERE weight >= 0.05 AND valid_to IS NULL")
            else:
                sql = ("SELECT source_id, target_id, weight FROM connections "
                       "WHERE weight >= 0.05")
            cur = conn.execute(sql)
            while True:
                rows = cur.fetchmany(50000)
                if not rows:
                    break
                for r in rows:
                    yield {
                        "source_id": r["source_id"],
                        "target_id": r["target_id"],
                        "weight": r["weight"],
                    }
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        """Add a REM bridge edge. H6: stamps bi-temporal metadata if schema supports it,
        so bridges are temporally traceable (`valid_from = ingestion_time = now`).
        """
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM connections "
                "WHERE (source_id = ? AND target_id = ?) "
                "OR (source_id = ? AND target_id = ?)",
                (source_id, target_id, target_id, source_id)
            ).fetchone()
            if not existing:
                now_ts = time.time()
                cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
                if {"ingestion_time", "valid_from"}.issubset(cols):
                    # H6: bi-temporal-aware insert
                    conn.execute(
                        """INSERT INTO connections
                           (source_id, target_id, weight, edge_type, created_at,
                            ingestion_time, valid_from)
                           VALUES (?, ?, ?, 'rem_bridge', ?, ?, ?)""",
                        (source_id, target_id, weight, now_ts, now_ts, now_ts),
                    )
                else:
                    # Pre-migration fallback
                    conn.execute(
                        "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at) "
                        "VALUES (?, ?, ?, 'rem_bridge', ?)",
                        (source_id, target_id, weight, now_ts)
                    )
                conn.commit()
        finally:
            conn.close()

    def prune_weak(self, threshold: float = 0.05) -> int:
        """H6: soft-delete weak edges (set valid_to=now) rather than hard-delete.

        Makes the connections table a temporal audit log — previous graph states
        remain queryable via `get_connections(..., at_time=past_ts)`.

        Only targets currently-valid edges (valid_to IS NULL). Already-expired
        edges are untouched. Falls back to hard-DELETE if bi-temporal columns
        aren't available (very old DBs pre-migration).
        """
        conn = self._connect()
        try:
            # Check if bi-temporal column exists
            cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
            if "valid_to" in cols:
                count = conn.execute(
                    """UPDATE connections SET valid_to = ?
                       WHERE weight < ? AND valid_to IS NULL""",
                    (time.time(), threshold),
                ).rowcount
            else:
                count = conn.execute(
                    "DELETE FROM connections WHERE weight < ?",
                    (threshold,)
                ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        # NM_DISABLE_CONN_HISTORY=1 is a substrate-pressure relief valve.
        # Sonnet investigation 2026-05-02 [verified-now]: zero production
        # code reads connection_history. With ~7.7M edges per NREM cycle,
        # the audit log was growing 11M+ rows/day for no consumer.
        if os.environ.get("NM_DISABLE_CONN_HISTORY") == "1":
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, old_weight, new_weight, reason, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def apply_connection_changes(
        self,
        updates: List[Tuple[int, int, float, float, str]],
        *,
        prune_threshold: float = 0.05,
    ) -> int:
        """Apply NREM edge updates in one SQLite transaction.

        D6 runtime proof showed the scheduled 03:45 dream cycle spending tens of
        minutes inside SQLite when NREM iterated millions of edges. The hotspot
        was per-edge connection churn: strengthen/weakens and history logging each
        opened a fresh SQLite connection and committed individually.

        Keep the fix narrow: batch only the existing per-edge update pattern so a
        whole NREM pass can use O(1) backend connections instead of O(N).
        """
        conn = self._connect()
        try:
            if updates:
                changed_at = time.time()
                conn.executemany(
                    "UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?",
                    [
                        (new_weight, source_id, target_id)
                        for source_id, target_id, _old_weight, new_weight, _reason in updates
                    ],
                )
                if os.environ.get("NM_DISABLE_CONN_HISTORY") != "1":
                    conn.executemany(
                        "INSERT INTO connection_history "
                        "(source_id, target_id, old_weight, new_weight, reason, changed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (source_id, target_id, old_weight, new_weight, reason, changed_at)
                            for source_id, target_id, old_weight, new_weight, reason in updates
                        ],
                    )

            cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
            if "valid_to" in cols:
                count = conn.execute(
                    """UPDATE connections SET valid_to = ?
                       WHERE weight < ? AND valid_to IS NULL""",
                    (time.time(), prune_threshold),
                ).rowcount
            else:
                count = conn.execute(
                    "DELETE FROM connections WHERE weight < ?",
                    (prune_threshold,),
                ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, insight_type, source_memory_id, content, confidence, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()

            insights = conn.execute(
                "SELECT insight_type, COUNT(*) FROM dream_insights GROUP BY insight_type"
            ).fetchall()

            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "insight_types": {r[0]: r[1] for r in insights},
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

def _detect_communities(edges, nodes, adj):
    """Community detection with Louvain preferred, BFS connected-components fallback.

    Louvain gives modularity-optimized partitions; BFS collapses all reachable
    nodes into one component, which degenerates once the auto-connect threshold
    has fired enough edges. Louvain is also deterministic given a seed.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for e in edges:
            w = e.get("weight", 1.0)
            if w and w > 0:
                g.add_edge(e["source_id"], e["target_id"], weight=float(w))
        if g.number_of_edges() == 0:
            return [[n] for n in nodes]
        parts = louvain_communities(g, weight="weight", seed=42)
        return [list(p) for p in parts]
    except Exception as exc:
        logger.debug("Louvain unavailable (%s); falling back to BFS components", exc)

    # BFS connected components fallback
    visited = set()
    out = []
    for node in nodes:
        if node in visited:
            continue
        component = []
        queue = [node]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            component.append(curr)
            for neighbor, _ in adj.get(curr, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        out.append(component)
    return out


class DreamEngine:
    """Autonomous background consolidation for neural memory.

    Three phases:
      NREM  — Replay recent memories, strengthen active, prune dead
      REM   — Explore isolated memories, discover bridges via embedding
      Insight — Community detection, bridge identification, abstraction
    """

    def __init__(
        self,
        backend: DreamBackend,
        neural_memory: Optional[Any] = None,
        idle_threshold: float = 300.0,     # 5 min idle
        memory_threshold: int = 50,         # dream every N new memories
        max_memories_per_cycle: int = 100,
    ):
        self._backend = backend
        self._memory = neural_memory        # NeuralMemory instance for think/recall
        self._idle_threshold = idle_threshold
        self._memory_threshold = memory_threshold
        self._max_memories = max_memories_per_cycle

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._memory_count_at_last_dream = 0
        self._dream_count = 0
        self._subprocess_jobs: Dict[str, Any] = {}

    @classmethod
    def sqlite(cls, db_path: str, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with SQLite backend."""
        backend = SQLiteDreamBackend(db_path)
        return cls(backend, neural_memory, **kwargs)

    @classmethod
    def mssql(cls, mssql_config: dict, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with MSSQL backend."""
        from dream_mssql_store import DreamMSSQLStore
        backend = DreamMSSQLStore.from_config(mssql_config)
        return cls(backend, neural_memory, **kwargs)

    def start(self) -> None:
        """Start the background dream daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dream_loop, daemon=True, name="dream-engine"
        )
        self._thread.start()
        logger.info(
            "Dream engine started: idle=%ss, threshold=%d",
            self._idle_threshold, self._memory_threshold,
        )

    def stop(self) -> None:
        """Stop the dream daemon."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Dream engine stopped after %d cycles", self._dream_count)

    def touch(self) -> None:
        """Signal activity — resets idle timer."""
        self._last_activity = time.time()

    def dream_now(self, dispatch: str = "inline") -> Dict[str, Any]:
        """Force an immediate dream cycle.

        H20: `dispatch` controls execution mode.
          - "inline" (default): runs in-process, blocks until complete. Original
            behavior preserved.
          - "subprocess": spawns a Python subprocess that re-opens the DB and
            runs the cycle there. Returns immediately with a job id; status
            file written to ~/.neural_memory/dream-jobs/<job_id>.json.
            Recall in the parent process is not blocked by the dream cycle.

        SQLite WAL mode (already enabled in SQLiteStore.__init__) makes
        concurrent read+write safe across processes.
        """
        if dispatch == "inline":
            return self._run_dream_cycle()
        elif dispatch == "subprocess":
            return self._dispatch_subprocess()
        else:
            raise ValueError(f"unknown dream dispatch: {dispatch!r}")

    def _dispatch_subprocess(self) -> Dict[str, Any]:
        """H20: spawn a subprocess to run the dream cycle. Returns immediately
        with a job descriptor; status file polled via dream_status().
        """
        import json as _json
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile
        from pathlib import Path as _Path

        # Locate the DB path so the subprocess can re-open it
        db_path = None
        if hasattr(self, "_memory") and self._memory is not None:
            store = getattr(self._memory, "store", None)
            db_path = getattr(store, "db_path", None) if store is not None else None
        # Fall back to a sane default if not introspectable
        if not db_path:
            db_path = str(_Path.home() / ".neural_memory" / "memory.db")

        job_id = f"dream-{int(time.time() * 1000)}-{_os.getpid()}"
        status_dir = _Path.home() / ".neural_memory" / "dream-jobs"
        status_dir.mkdir(parents=True, exist_ok=True)
        status_path = status_dir / f"{job_id}.json"
        status_path.write_text(_json.dumps({
            "job_id": job_id,
            "status": "queued",
            "started_at": time.time(),
        }))

        # Locate the python/ dir so the subprocess can `import memory_client`
        # & `import dream_engine`. __file__ lives in python/ — use its parent.
        py_dir = str(_Path(__file__).resolve().parent)

        runner = (
            "import sys, json, time\n"
            f"sys.path.insert(0, {py_dir!r})\n"
            "import os\n"
            "from memory_client import NeuralMemory\n"
            "from dream_engine import DreamEngine\n"
            "embedding_backend = os.environ.get('NEURAL_MEMORY_EMBED_BACKEND', 'auto')\n"
            f"mem = NeuralMemory(db_path={db_path!r}, embedding_backend=embedding_backend, use_cpp=False, use_hnsw=False)\n"
            f"engine = DreamEngine.sqlite({db_path!r}, neural_memory=mem)\n"
            "started = time.time()\n"
            "try:\n"
            "    result = engine._run_dream_cycle()\n"
            "    status = 'complete'\n"
            "    error = None\n"
            "except Exception as exc:\n"
            "    result = {}\n"
            "    status = 'error'\n"
            "    error = str(exc)\n"
            "try:\n"
            "    mem.close()\n"
            "except Exception as close_exc:\n"
            "    if error is None:\n"
            "        error = 'close failed: ' + str(close_exc)\n"
            "        status = 'error'\n"
            f"sp = {str(status_path)!r}\n"
            "with open(sp, 'w') as fh:\n"
            "    json.dump({\n"
            f"        'job_id': {job_id!r},\n"
            "        'status': status,\n"
            f"        'started_at': started,\n"
            "        'finished_at': time.time(),\n"
            "        'result': result,\n"
            "        'error': error,\n"
            "    }, fh)\n"
        )

        # stdout/stderr to /dev/null so subprocess doesn't pipe back
        proc = _subprocess.Popen(
            [_sys.executable, "-c", runner],
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
            close_fds=True,
        )
        self._subprocess_jobs[job_id] = proc
        return {
            "job_id": job_id,
            "pid": proc.pid,
            "status_path": str(status_path),
            "status": "queued",
            "dispatch": "subprocess",
        }

    def dream_status(self, job_id: str) -> Dict[str, Any]:
        """H20: poll job status from the status file written by the subprocess.

        Returns {'status': 'unknown'} if the job_id isn't found.
        Status values: 'queued' | 'complete' | 'error' | 'unknown'.
        """
        from pathlib import Path as _Path
        import json as _json
        status_path = _Path.home() / ".neural_memory" / "dream-jobs" / f"{job_id}.json"
        if not status_path.exists():
            return {"job_id": job_id, "status": "unknown"}
        try:
            status = _json.loads(status_path.read_text())
            if status.get("status") in {"complete", "error"}:
                proc = self._subprocess_jobs.get(job_id)
                if proc is not None:
                    try:
                        proc.wait(timeout=5.0)
                    except Exception:
                        pass
                    if proc.poll() is not None:
                        self._subprocess_jobs.pop(job_id, None)
            return status
        except Exception as exc:
            return {"job_id": job_id, "status": "error", "error": str(exc)}

    # -- Main loop -----------------------------------------------------------

    def _dream_loop(self) -> None:
        """Background daemon: dream when idle or threshold reached."""
        while self._running:
            try:
                time.sleep(30)
                if not self._running:
                    break

                idle = time.time() - self._last_activity
                try:
                    stats = self._memory.stats() if self._memory else {"memories": 0}
                    total = stats.get("memories", 0)
                except Exception:
                    total = 0
                new_since_last = total - self._memory_count_at_last_dream

                should_dream = (
                    idle >= self._idle_threshold
                    or new_since_last >= self._memory_threshold
                )

                if should_dream:
                    logger.info(
                        "Dream cycle triggered: idle=%.0fs, new=%d",
                        idle, new_since_last,
                    )
                    self._run_dream_cycle()

            except Exception as e:
                logger.debug("Dream loop error: %s", e)
                time.sleep(60)

    # -- Dream Cycle ---------------------------------------------------------

    def _run_dream_cycle(self) -> Dict[str, Any]:
        """Execute a full NREM → REM → Insight cycle."""
        with self._lock:
            start = time.time()
            total_stats: Dict[str, Any] = {
                "nrem": {},
                "rem": {},
                "insights": {},
                "self_image": {},
            }

            try:
                total_stats["nrem"] = self._phase_nrem()
                total_stats["rem"] = self._phase_rem()
                total_stats["insights"] = self._phase_insights()
                total_stats["self_image"] = self._phase_self_image()

                self._dream_count += 1
                if self._memory:
                    try:
                        s = self._memory.stats()
                        self._memory_count_at_last_dream = s.get("memories", 0)
                    except Exception:
                        pass

                total_stats["duration"] = time.time() - start
                total_stats["dream_id"] = self._dream_count

                logger.info(
                    "Dream #%d complete: %.1fs | NREM: %d+/ %d- / %d pruned | REM: %d bridges | Insights: %d | Self-image: %d",
                    self._dream_count, total_stats["duration"],
                    total_stats["nrem"].get("strengthened", 0),
                    total_stats["nrem"].get("weakened", 0),
                    total_stats["nrem"].get("pruned", 0),
                    total_stats["rem"].get("bridges", 0),
                    total_stats["insights"].get("insights", 0),
                    total_stats["self_image"].get("insights", 0),
                )

            except Exception as e:
                logger.error("Dream cycle failed: %s", e)
                total_stats["error"] = str(e)

            return total_stats

    # -- Phase 1: NREM -------------------------------------------------------

    def _phase_nrem(self) -> Dict[str, Any]:
        """NREM: Replay, strengthen active, weaken inactive, prune dead.

        For each recent memory:
          1. Fire spreading activation
          2. Activated edges: weight += 0.05
          3. Non-activated edges: weight -= 0.01
          4. Edges below 0.05: prune
        """
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}
        session_id = self._backend.start_session("nrem")

        try:
            memories = self._backend.get_recent_memories(self._max_memories)
            if not memories:
                return stats

            activated_edges = set()

            for mem in memories:
                mid = mem["id"]
                if self._memory:
                    try:
                        activated = self._memory.think(mid, depth=2)
                        for a in activated:
                            aid = a.get("id")
                            if aid and aid != mid:
                                activated_edges.add((min(mid, aid), max(mid, aid)))
                    except Exception:
                        pass
                stats["processed"] += 1

            # Get all connections and update weights. Stream via
            # iter_connections() — fetchall() previously pulled all 7.7M
            # active edges into a 6.7GB Python heap before any work began.
            updates: List[Tuple[int, int, float, float, str]] = []
            try:
                update_batch_size = int(os.environ.get("NM_NREM_UPDATE_BATCH", "50000"))
            except ValueError:
                update_batch_size = 50000
            update_batch_size = max(1, update_batch_size)
            can_batch_updates = hasattr(self._backend, "apply_connection_changes")

            def flush_updates(*, prune_threshold: float = 0.0) -> int:
                # In-flight calls pass prune_threshold=0.0 — the prune SQL
                # `WHERE weight < 0.0` matches nothing, so it's a no-op.
                # Only the final flush passes 0.05 to actually prune.
                nonlocal updates
                if not can_batch_updates:
                    return 0
                pruned = self._backend.apply_connection_changes(
                    updates,
                    prune_threshold=prune_threshold,
                )
                updates = []
                return pruned

            for conn in self._backend.iter_connections():
                src, tgt = conn["source_id"], conn["target_id"]
                key = (min(src, tgt), max(src, tgt))

                if key in activated_edges:
                    old_w = conn["weight"]
                    updates.append((src, tgt, old_w, min(old_w + 0.05, 1.0), "nrem_strengthen"))
                    stats["strengthened"] += 1
                else:
                    old_w = conn["weight"]
                    if old_w > 0.05:
                        updates.append((src, tgt, old_w, max(old_w - 0.01, 0.0), "nrem_weaken"))
                        stats["weakened"] += 1
                if can_batch_updates and len(updates) >= update_batch_size:
                    flush_updates()

            # D6 runtime proof on the live 03:45 scheduler showed that opening a
            # fresh SQLite connection + commit per edge turns NREM into a giant
            # SQLite hot loop once the graph reaches millions of connections.
            # Batch the exact same edge updates into one backend transaction when
            # the backend supports it, bounded by NM_NREM_UPDATE_BATCH so NREM no
            # longer holds a full-graph update list in memory. Preserve the old
            # step-by-step path for other backends.
            if can_batch_updates:
                stats["pruned"] = flush_updates(prune_threshold=0.05)
            else:
                for src, tgt, old_w, new_w, reason in updates:
                    delta = new_w - old_w
                    if delta > 0:
                        self._backend.strengthen_connection(src, tgt, delta)
                    else:
                        self._backend.weaken_connection(src, tgt, abs(delta))
                    self._backend.log_connection_change(src, tgt, old_w, new_w, reason)
                # Prune dead connections
                stats["pruned"] = self._backend.prune_weak(0.05)

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("NREM phase error: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def _phase_rem(self) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Search via embedding similarity for unconnected but similar
        3. Create tentative bridge connections (weight 0.1-0.3)
        """
        stats = {"explored": 0, "bridges": 0, "rejected": 0}
        session_id = self._backend.start_session("rem")

        try:
            isolated = self._backend.get_isolated_memories(max_connections=3, limit=50)
            if not isolated:
                return stats

            for mem in isolated:
                mid = mem["id"]
                content = mem["content"]

                if not self._memory or not content:
                    continue

                try:
                    similar = self._memory.recall(content[:200], k=10)
                    stats["explored"] += 1

                    for sim in similar:
                        sim_id = sim.get("id")
                        sim_score = sim.get("similarity", 0)

                        if not sim_id or sim_id == mid:
                            continue
                        if sim_score < 0.3 or sim_score > 0.95:
                            continue

                        bridge_weight = round(sim_score * 0.3, 3)
                        self._backend.add_bridge(mid, sim_id, bridge_weight)
                        self._backend.log_connection_change(
                            mid, sim_id, 0.0, bridge_weight, "rem_bridge"
                        )
                        stats["bridges"] += 1

                except Exception:
                    pass

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("REM phase error: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def _phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction.

        1. Find connected components (communities)
        2. Identify bridge nodes connecting communities
        3. Create insight memories for dense clusters
        """
        stats = {"communities": 0, "bridges": 0, "insights": 0}
        session_id = self._backend.start_session("insight")

        try:
            # Stream once into adjacency; community detection still needs the
            # full edge list, but we avoid the second materialization that
            # the previous (edges = get_connections() then iterate) pattern
            # implied.
            edges: List[Dict[str, Any]] = []
            adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            nodes = set()
            for e in self._backend.iter_connections():
                edges.append(e)
                s, t, w = e["source_id"], e["target_id"], e["weight"]
                adj[s].append((t, w))
                adj[t].append((s, w))
                nodes.add(s)
                nodes.add(t)
            if not edges:
                return stats

            # Community detection: Louvain (modularity-optimized) with BFS fallback.
            # Louvain finds denser sub-structure inside connected components, which
            # matters as the graph gets well-connected (connected-components collapses
            # everything into one blob once the auto-connect threshold fires enough).
            communities = _detect_communities(edges, nodes, adj)
            stats["communities"] = len(communities)

            # Map nodes to communities
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # Find bridge nodes
            bridge_nodes = set()
            for e in edges:
                s_comm = node_to_comm.get(e["source_id"], -1)
                t_comm = node_to_comm.get(e["target_id"], -1)
                if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                    bridge_nodes.add(e["source_id"])
                    bridge_nodes.add(e["target_id"])
            stats["bridges"] = len(bridge_nodes)

            # Create cluster insights. D5: keep legacy dream_insights analytics rows,
            # but also promote the logical insight into the retrievable memories
            # surface via kind='dream_insight' + summarizes evidence edges.
            for i, comm in enumerate(communities):
                comm_ids = [
                    int(mid) for mid in comm
                    if self._memory_kind(int(mid)) not in {"dream_insight", "entity", "locus"}
                ]
                if len(comm_ids) < 2:
                    continue
                self._create_retrievable_insight(comm_ids)
                theme = self._extract_theme(comm_ids)
                confidence = min(len(comm_ids) / 10.0, 1.0)
                content = f"Cluster of {len(comm_ids)} related memories: {theme}"
                self._backend.add_insight(session_id, "cluster", comm_ids[0], content, confidence)
                stats["insights"] += 1

            # Create bridge insights
            for bnode in bridge_nodes:
                bridging_communities = set()
                for e in edges:
                    if e["source_id"] == bnode or e["target_id"] == bnode:
                        other = e["target_id"] if e["source_id"] == bnode else e["source_id"]
                        bridging_communities.add(node_to_comm.get(other, -1))
                bridging_communities.discard(-1)

                if len(bridging_communities) >= 2:
                    if self._memory_kind(int(bnode)) in {"dream_insight", "entity", "locus"}:
                        continue
                    self._create_retrievable_insight([int(bnode)])
                    content = (
                        f"Bridge connecting {len(bridging_communities)} communities, "
                        f"memory #{bnode}"
                    )
                    self._backend.add_insight(session_id, "bridge", bnode, content, 0.8)
                    stats["insights"] += 1

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("Insight phase error: %s", e)

        return stats

    def _phase_self_image(self) -> Dict[str, Any]:
        """D5: synthesize self-image observations into retrievable insight nodes.

        This phase is intentionally local-only: it reuses the existing
        dream_insights table for analytics and NeuralMemory.create_insight_from_cluster()
        for the production retrieval surface. No schema changes, no new storage API.
        """
        stats = {"processed": 0, "candidates": 0, "insights": 0}
        session_id = self._backend.start_session("self_image")

        try:
            candidates = []
            seen = set()
            for mem in self._backend.get_recent_memories(self._max_memories):
                mid = int(mem.get("id") or 0)
                content = mem.get("content") or ""
                if not mid or mid in seen:
                    continue
                if self._memory_kind(mid) in {"dream_insight", "entity", "locus"}:
                    continue
                stats["processed"] += 1
                if self._is_self_image_candidate(content):
                    candidates.append({"id": mid, "content": content})
                    seen.add(mid)
                if len(candidates) >= 8:
                    break

            stats["candidates"] = len(candidates)
            if len(candidates) < 2:
                self._backend.finish_session(session_id, stats)
                return stats

            source_ids = [c["id"] for c in candidates]
            self._create_retrievable_insight(source_ids)
            theme = self._extract_theme(source_ids)
            content = (
                f"Self-image insight from {len(source_ids)} operator/runtime memories: {theme}"
            )
            confidence = min(0.4 + (len(source_ids) * 0.1), 0.9)
            self._backend.add_insight(
                session_id,
                "self_image",
                source_ids[0],
                content,
                confidence,
            )
            stats["insights"] = 1
            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("Self-image phase error: %s", e)

        return stats

    # -- Helpers -------------------------------------------------------------

    def _create_retrievable_insight(self, memory_ids: List[int]) -> int:
        """Create a Phase-7 dream_insight memory node with summarizes edges.

        Legacy dream_insights rows are not queried by recall; this helper bridges
        dream synthesis into the actual NeuralMemory retrieval surface while
        staying schema-clean. If an existing dream_insight already summarizes all
        source ids, reuse it instead of creating another node every dream cycle.
        """
        source_ids = []
        seen = set()
        for mid in memory_ids:
            try:
                mid_i = int(mid)
            except Exception:
                continue
            if mid_i > 0 and mid_i not in seen:
                source_ids.append(mid_i)
                seen.add(mid_i)
        if not source_ids:
            return 0
        if not self._memory or not hasattr(self._memory, "create_insight_from_cluster"):
            return 0

        existing = self._existing_retrievable_insight(source_ids)
        if existing:
            return existing
        try:
            return int(self._memory.create_insight_from_cluster(source_ids) or 0)
        except Exception as exc:
            logger.debug("create_insight_from_cluster failed: %s", exc)
            return 0

    def _existing_retrievable_insight(self, source_ids: List[int]) -> int:
        """Return an existing dream_insight node that summarizes all sources."""
        if not self._memory or not hasattr(self._memory, "store"):
            return 0
        store = getattr(self._memory, "store", None)
        conn = getattr(store, "conn", None)
        lock = getattr(store, "_lock", None)
        if conn is None or lock is None or not source_ids:
            return 0
        placeholders = ",".join("?" * len(source_ids))
        sql = f"""
            SELECT c.source_id
            FROM connections c
            JOIN memories m ON m.id = c.source_id
            WHERE m.kind = 'dream_insight'
              AND c.edge_type = 'summarizes'
              AND c.target_id IN ({placeholders})
            GROUP BY c.source_id
            HAVING COUNT(DISTINCT c.target_id) >= ?
            LIMIT 1
        """
        try:
            with lock:
                row = conn.execute(sql, tuple(source_ids) + (len(set(source_ids)),)).fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            logger.debug("retrievable insight lookup failed: %s", exc)
            return 0

    def _memory_kind(self, memory_id: int) -> str:
        """Best-effort kind lookup for filtering overlay/self-generated nodes."""
        if not self._memory or not hasattr(self._memory, "store"):
            return ""
        store = getattr(self._memory, "store", None)
        conn = getattr(store, "conn", None)
        lock = getattr(store, "_lock", None)
        if conn is None or lock is None:
            return ""
        try:
            with lock:
                row = conn.execute("SELECT kind FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return (row[0] or "") if row else ""
        except Exception:
            return ""

    @staticmethod
    def _is_self_image_candidate(content: str) -> bool:
        """True when content describes the agent/system observing its own work."""
        text = (content or "").lower()
        if not text:
            return False
        actor_terms = (
            "valiendo", "hermes", "claude code", "claude-code",
            "operator canon", "runtime proof", "bridge ack", "i acked",
            "i verified", "i patched", "my lane", "self-image",
        )
        action_terms = (
            "verified", "patched", "ack", "checkpoint", "bridge",
            "runtime", "canon", "handoff", "boundary", "proof", "no-send",
        )
        return any(term in text for term in actor_terms) and any(term in text for term in action_terms)

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs (simple keyword frequency)."""
        # If we have memory access, get contents via store (thread-safe)
        contents = []
        if self._memory and hasattr(self._memory, 'store'):
            try:
                placeholders = ",".join("?" * len(node_ids))
                with self._memory.store._lock:
                    rows = self._memory.store.conn.execute(
                        f"SELECT content FROM memories WHERE id IN ({placeholders})",
                        tuple(node_ids)
                    ).fetchall()
                    contents = [r[0] for r in rows if r[0]]
            except Exception:
                pass

        if not contents:
            return f"{len(node_ids)} memories"

        word_freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "it", "its", "this", "that", "user", "assistant", "and", "or",
            "but", "not", "if", "then", "so", "just", "also", "very",
            "really", "like", "get", "got", "want", "need", "think",
            "know", "see", "look", "make", "let", "use", "still",
        }
        for c in contents:
            for w in c.lower().split():
                w = w.strip(".,!?;:'\"()[]{}#@")
                if len(w) > 3 and w not in stopwords:
                    word_freq[w] += 1

        top = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
        return ", ".join(w for w, _ in top) if top else "mixed topics"

    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        base = self._backend.get_dream_stats()
        base["engine_running"] = self._running
        base["dream_cycles"] = self._dream_count
        return base
