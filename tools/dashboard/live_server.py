#!/usr/bin/env python3
"""
live_server.py — FastAPI WebSocket server for Neural Memory Live Dashboard.

Auto-detects MSSQL (primary) vs SQLite (fallback). Reads config from
~/.hermes/config.yaml for MSSQL credentials.

Usage:
    python live_server.py                             # auto-detect DB
    python live_server.py --port 8443                 # custom port
    python live_server.py --db /path/to/memory.db     # force SQLite
    python live_server.py --no-tls                    # HTTP only
    python live_server.py --desktop-layer             # reduced logging
    python live_server.py --watch-interval 1          # poll every 1s
"""
import argparse
import asyncio
import fcntl
import json
import logging
import os
import pty
import struct
import sys
import termios
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger("neural-live")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
TEMPLATE_DIR   = Path(__file__).parent
POLL_INTERVAL  = 2.0
CONFIG_PATH    = os.path.expanduser("~/.hermes/config.yaml")
NODE_LIMIT     = 100   # top hub nodes to visualise
EDGE_LIMIT     = 500   # max edges (by weight, descending)

_DB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-worker")

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
# Config loader
# ═══════════════════════════════════════════════════════════

def load_mssql_config() -> dict | None:
    try:
        import yaml
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        mssql = cfg.get("memory", {}).get("neural", {}).get("dream", {}).get("mssql", {})
        if not mssql.get("password"):
            return None
        return {
            "server":   mssql.get("server",   "127.0.0.1"),
            "database": mssql.get("database", "NeuralMemory"),
            "username": mssql.get("username", "SA"),
            "password": mssql["password"],
            "driver":   mssql.get("driver",   "{ODBC Driver 18 for SQL Server}"),
        }
    except Exception as exc:
        logger.warning(f"Cannot read MSSQL config: {exc}")
        return None


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
# Data readers
# ═══════════════════════════════════════════════════════════

def read_sqlite(db_path: str) -> dict:
    import sqlite3
    t0 = time.perf_counter()
    conn = sqlite3.connect(db_path, timeout=10)
    cur  = conn.cursor()

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
    placeholders = ",".join(str(i) for i in hub_ids)  # ints from our own query — safe
    cur.execute(
        f"SELECT source_id, target_id, weight FROM connections "
        f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) "
        f"ORDER BY weight DESC LIMIT {EDGE_LIMIT}"
    )
    edges = [{"source": r[0], "target": r[1], "weight": round(r[2], 4)} for r in cur.fetchall()]

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

    conn.close()
    ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["db_query_ms"] = ms
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "dream_sessions": dream_sessions,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": actual_dim, "source": "SQLite",
            "path": db_path, "query_ms": ms,
        },
    }


def read_mssql(mssql_cfg: dict) -> dict:
    import pyodbc
    t0 = time.perf_counter()
    cs = (
        f"DRIVER={mssql_cfg['driver']};"
        f"SERVER={mssql_cfg['server']};"
        f"DATABASE={mssql_cfg['database']};"
        f"UID={mssql_cfg['username']};PWD={mssql_cfg['password']};"
        "TrustServerCertificate=yes;Encrypt=no;"
    )
    conn = pyodbc.connect(cs, autocommit=True)
    cur  = conn.cursor()

    cur.execute("""
        SELECT TOP 200 m.id, m.label, LEN(ISNULL(m.content,'')) AS clen,
               m.salience, m.access_count,
               ISNULL(o.out_degree, 0), ISNULL(i.in_degree, 0), ISNULL(o.avg_weight, 0)
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree,
                          AVG(CAST(weight AS FLOAT)) AS avg_weight
                   FROM connections GROUP BY source_id) o ON m.id = o.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) i ON m.id = i.target_id
        ORDER BY (ISNULL(o.out_degree,0) + ISNULL(i.in_degree,0)) DESC
    """)
    nodes = []
    for r in cur.fetchall():
        lbl = r[1] or ""
        nodes.append({
            "id": r[0], "label": lbl[:50], "category": _categorize(lbl),
            "content_length": r[2] or 0, "salience": r[3] or 1.0,
            "access_count": r[4] or 0, "out_degree": r[5],
            "in_degree": r[6], "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })

    hub_ids = [n["id"] for n in nodes[:NODE_LIMIT]]
    id_list = ",".join(str(x) for x in hub_ids)
    if id_list:
        cur.execute(
            f"SELECT TOP {EDGE_LIMIT} source_id, target_id, weight FROM connections "
            f"WHERE source_id IN ({id_list}) AND target_id IN ({id_list}) "
            f"ORDER BY weight DESC"
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
        ) t GROUP BY cat ORDER BY COUNT(*) DESC
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
        ) t GROUP BY bucket
    """)
    weights = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) FROM memories");    n_mem  = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections"); n_conn = cur.fetchone()[0]

    n_dreams = 0
    try:
        cur.execute("SELECT COUNT(*) FROM dream_sessions"); n_dreams = cur.fetchone()[0]
    except Exception:
        pass

    cur.execute("SELECT TOP 1 vector_dim FROM memories WHERE embedding IS NOT NULL")
    dr  = cur.fetchone()
    dim = dr[0] if dr else 1024

    conn.close()
    ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["db_query_ms"] = ms
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": dim, "source": "MSSQL",
            "path": f"{mssql_cfg['server']}/{mssql_cfg['database']}",
            "dream_sessions": n_dreams, "query_ms": ms,
        },
    }


def load_data(args) -> tuple[dict, callable]:
    mssql_cfg = None if args.db else load_mssql_config()
    if mssql_cfg:
        try:
            data = read_mssql(mssql_cfg)
            s = data["stats"]
            logger.info(f"MSSQL: {s['memories']}M  {s['connections']}C  ({s['query_ms']}ms)")
            _metrics["db_source"] = "MSSQL"
            return data, lambda: read_mssql(mssql_cfg)
        except Exception as exc:
            logger.warning(f"MSSQL failed ({exc}), falling back to SQLite")

    db_path = args.db or DEFAULT_SQLITE
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    data = read_sqlite(db_path)
    s = data["stats"]
    logger.info(f"SQLite: {s['memories']}M  {s['connections']}C  ({s['query_ms']}ms)")
    _metrics["db_source"] = "SQLite"
    return data, lambda: read_sqlite(db_path)


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


def compute_data_hash(data: dict) -> str:
    s = data["stats"]
    return f"{s['memories']}:{s['connections']}"


async def poll_db_changes():
    global current_data, current_hash
    loop = asyncio.get_event_loop()
    while True:
        try:
            _metrics["poll_count"] += 1
            new_data = await loop.run_in_executor(_DB_EXECUTOR, reload_fn)
            new_hash = compute_data_hash(new_data)
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
    task = asyncio.create_task(poll_db_changes())
    yield
    task.cancel()
    _DB_EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Neural Memory Live Dashboard", lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(_build_html("template-live.html"))


@app.get("/desktop")
async def desktop():
    return HTMLResponse(_build_html("template-desktop.html"))


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

    parser = argparse.ArgumentParser(description="Neural Memory Live Dashboard Server")
    parser.add_argument("--db",             default=None,  help="Force SQLite path")
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
                "-subj", "/CN=neural-memory-dashboard/O=NeuralMemory/C=DE",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0",
            ], capture_output=True, check=True)
        ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}

    proto = "https" if not args.no_tls else "http"
    ws_proto = proto.replace("http", "ws")
    s = current_data["stats"]

    print(f"\n{'═' * 64}")
    print(f"  ◈  Neural Memory — LIVE Dashboard  [{s['source']}]")
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
