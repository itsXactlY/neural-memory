#!/usr/bin/env python3
"""
live_server.py — FastAPI WebSocket server for Neural Memory Live Dashboard.

Auto-detects MSSQL (primary) vs SQLite (fallback). Reads config from
~/.hermes/config.yaml for MSSQL credentials.

Usage:
    python live_server.py                          # auto-detect DB
    python live_server.py --port 8443              # custom port
    python live_server.py --db /path/to/memory.db  # force SQLite
    python live_server.py --no-tls                 # HTTP only
    python live_server.py --desktop-layer          # reduced logging
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
import hashlib
import logging
from pathlib import Path
from contextlib import asynccontextmanager

logger = logging.getLogger("neural-live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
TEMPLATE_DIR = Path(__file__).parent
POLL_INTERVAL = 3.0
EMBEDDING_DIM = 384
CONFIG_PATH = os.path.expanduser("~/.hermes/config.yaml")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    print("FastAPI + uvicorn required: pip install fastapi uvicorn")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Config loader
# ═══════════════════════════════════════════════════════════

def load_mssql_config():
    """Read MSSQL credentials from ~/.hermes/config.yaml."""
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        mssql = cfg.get("memory", {}).get("neural", {}).get("dream", {}).get("mssql", {})
        if not mssql.get("password"):
            return None
        return {
            "server": mssql.get("server", "127.0.0.1"),
            "database": mssql.get("database", "NeuralMemory"),
            "username": mssql.get("username", "SA"),
            "password": mssql["password"],
            "driver": mssql.get("driver", "{ODBC Driver 18 for SQL Server}"),
        }
    except Exception as e:
        logger.warning(f"Cannot read MSSQL config: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# Categorize
# ═══════════════════════════════════════════════════════════

def _categorize(label: str) -> str:
    if label.startswith("peer:"):
        return "Peer"
    if label.startswith(("turn:", "msg:")):
        return "Conversation"
    if label.startswith("session:"):
        return "Session"
    if label.startswith("doc:"):
        return "Document"
    if label.startswith("skill:"):
        return "Skill"
    return "Other"


# ═══════════════════════════════════════════════════════════
# Data readers
# ═══════════════════════════════════════════════════════════

def read_sqlite(db_path: str) -> dict:
    """Read graph data from SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT m.id, m.label, m.content, m.salience, m.access_count,
               COALESCE(out_d.out_degree, 0) AS out_degree,
               COALESCE(in_d.in_degree, 0) AS in_degree,
               COALESCE(out_d.avg_weight, 0) AS avg_weight
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree, AVG(weight) AS avg_weight
                   FROM connections GROUP BY source_id) out_d ON m.id = out_d.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) in_d ON m.id = in_d.target_id
        ORDER BY (COALESCE(out_d.out_degree,0) + COALESCE(in_d.in_degree,0)) DESC
    """)
    nodes = []
    for r in cur.fetchall():
        label = r[1] or ""
        nodes.append({
            "id": r[0], "label": label[:50], "category": _categorize(label),
            "content_length": len(r[2]) if r[2] else 0,
            "salience": r[3] or 1.0, "access_count": r[4] or 0,
            "out_degree": r[5], "in_degree": r[6], "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })
    hub_ids = [n["id"] for n in nodes[:120]]
    id_set = set(hub_ids)
    cur.execute("SELECT source_id, target_id, weight FROM connections")
    edges = []
    for r in cur.fetchall():
        if r[0] in id_set and r[1] in id_set:
            edges.append({"source": r[0], "target": r[1], "weight": round(r[2], 4)})
    cur.execute("""
        SELECT cat, COUNT(*) FROM (
            SELECT CASE
                WHEN label LIKE 'peer:%' THEN 'Peer'
                WHEN label LIKE 'turn:%' OR label LIKE 'msg:%' THEN 'Conversation'
                WHEN label LIKE 'session:%' THEN 'Session'
                WHEN label LIKE 'doc:%' THEN 'Document'
                WHEN label LIKE 'skill:%' THEN 'Skill'
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
    cur.execute("SELECT COUNT(*) FROM memories")
    n_mem = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections")
    n_conn = cur.fetchone()[0]
    conn.close()
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": EMBEDDING_DIM, "source": "SQLite",
            "path": db_path,
        },
    }


def read_mssql(mssql_cfg: dict) -> dict:
    """Read graph data from MSSQL."""
    import pyodbc
    conn_str = (
        f"DRIVER={mssql_cfg['driver']};"
        f"SERVER={mssql_cfg['server']};"
        f"DATABASE={mssql_cfg['database']};"
        f"UID={mssql_cfg['username']};PWD={mssql_cfg['password']};"
        f"TrustServerCertificate=yes;Encrypt=no;"
    )
    conn = pyodbc.connect(conn_str, autocommit=True)
    cur = conn.cursor()

    cur.execute("""
        SELECT TOP 400 m.id, m.label, LEN(ISNULL(m.content,'')) AS clen,
               m.salience, m.access_count,
               ISNULL(o.out_degree, 0), ISNULL(i.in_degree, 0),
               ISNULL(o.avg_weight, 0)
        FROM memories m
        LEFT JOIN (SELECT source_id, COUNT(*) AS out_degree, AVG(CAST(weight AS FLOAT)) AS avg_weight
                   FROM connections GROUP BY source_id) o ON m.id = o.source_id
        LEFT JOIN (SELECT target_id, COUNT(*) AS in_degree
                   FROM connections GROUP BY target_id) i ON m.id = i.target_id
        ORDER BY (ISNULL(o.out_degree,0) + ISNULL(i.in_degree,0)) DESC
    """)
    nodes = []
    for r in cur.fetchall():
        label = r[1] or ""
        nodes.append({
            "id": r[0], "label": label[:50], "category": _categorize(label),
            "content_length": r[2] or 0, "salience": r[3] or 1.0,
            "access_count": r[4] or 0, "out_degree": r[5],
            "in_degree": r[6], "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })

    hub_ids = [n["id"] for n in nodes[:120]]
    id_set = set(hub_ids)
    id_list = ",".join(str(x) for x in hub_ids)
    if id_list:
        cur.execute(f"SELECT source_id, target_id, weight FROM connections WHERE source_id IN ({id_list}) AND target_id IN ({id_list})")
        edges = [{"source": r[0], "target": r[1], "weight": round(r[2], 4)} for r in cur.fetchall()]
    else:
        edges = []

    cur.execute("""
        SELECT cat, COUNT(*) FROM (
            SELECT CASE
                WHEN label LIKE 'peer:%' THEN 'Peer'
                WHEN label LIKE 'turn:%' OR label LIKE 'msg:%' THEN 'Conversation'
                WHEN label LIKE 'session:%' THEN 'Session'
                WHEN label LIKE 'doc:%' THEN 'Document'
                WHEN label LIKE 'skill:%' THEN 'Skill'
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

    cur.execute("SELECT COUNT(*) FROM memories")
    n_mem = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections")
    n_conn = cur.fetchone()[0]

    # Dream sessions
    try:
        cur.execute("SELECT COUNT(*) FROM dream_sessions")
        n_dreams = cur.fetchone()[0]
    except:
        n_dreams = 0

    conn.close()
    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "stats": {
            "memories": n_mem, "connections": n_conn,
            "embedding_dim": EMBEDDING_DIM, "source": "MSSQL",
            "path": f"{mssql_cfg['server']}/{mssql_cfg['database']}",
            "dream_sessions": n_dreams,
        },
    }


def load_data(args) -> tuple[dict, callable]:
    """Auto-detect DB backend. Returns (data, reload_fn)."""
    mssql_cfg = None if args.db else load_mssql_config()

    if mssql_cfg:
        try:
            data = read_mssql(mssql_cfg)
            logger.info(f"MSSQL: {data['stats']['memories']} memories, {data['stats']['connections']} connections")
            return data, lambda: read_mssql(mssql_cfg)
        except Exception as e:
            logger.warning(f"MSSQL failed ({e}), falling back to SQLite")

    db_path = args.db or DEFAULT_SQLITE
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    data = read_sqlite(db_path)
    logger.info(f"SQLite: {data['stats']['memories']} memories, {data['stats']['connections']} connections")
    return data, lambda: read_sqlite(db_path)


# ═══════════════════════════════════════════════════════════
# WebSocket connection manager
# ═══════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


manager = ConnectionManager()
current_data: dict = {}
current_hash: str = ""
reload_fn: callable = None


def compute_data_hash(data: dict) -> str:
    return f"{data['stats']['memories']}:{data['stats']['connections']}"


async def poll_db_changes():
    """Poll DB for changes and broadcast updates."""
    global current_data, current_hash
    while True:
        try:
            new_data = reload_fn()
            new_hash = compute_data_hash(new_data)
            if new_hash != current_hash:
                old_n_mem = current_data.get("stats", {}).get("memories", 0)
                old_n_conn = current_data.get("stats", {}).get("connections", 0)
                delta = {
                    "type": "update",
                    "stats": new_data["stats"],
                    "delta": {
                        "new_memories": new_data["stats"]["memories"] - old_n_mem,
                        "new_connections": new_data["stats"]["connections"] - old_n_conn,
                    },
                    "nodes": new_data["nodes"],
                    "edges": new_data["edges"],
                    "categories": new_data["categories"],
                    "weights": new_data["weights"],
                }
                current_data = new_data
                current_hash = new_hash
                if manager.connections:
                    await manager.broadcast(delta)
                    logger.info(f"Pushed update: {current_data['stats']['memories']}M {current_data['stats']['connections']}C")
        except Exception as e:
            logger.error(f"DB poll error: {e}")
        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_data, current_hash, reload_fn
    task = asyncio.create_task(poll_db_changes())
    yield
    task.cancel()


app = FastAPI(title="Neural Memory Live Dashboard", lifespan=lifespan)


@app.get("/")
async def index():
    template_path = TEMPLATE_DIR / "template-live.html"
    if not template_path.exists():
        template_path = TEMPLATE_DIR / "template.html"
    html = template_path.read_text()
    libs_dir = TEMPLATE_DIR / ".lib_cache"
    plotly_js = ""
    threejs_js = ""
    forcegraph_js = ""
    pf = libs_dir / "plotly-2.27.0.min.js"
    if pf.exists():
        plotly_js = f"<script>{pf.read_text()}</script>"
    tf = libs_dir / "three-0.160.0.min.js"
    if tf.exists():
        threejs_js = f"<script>{tf.read_text()}</script>"
    ff = libs_dir / "3d-force-graph.min.js"
    if ff.exists():
        forcegraph_js = f"<script>{ff.read_text()}</script>"
    html = html.replace("__PLOTLY_SCRIPT__", plotly_js)
    html = html.replace("__THREEJS_SCRIPT__", threejs_js)
    html = html.replace("__FORCEGRAPH_SCRIPT__", forcegraph_js)
    html = html.replace("__DATA_JSON__", json.dumps(current_data))
    return HTMLResponse(content=html)

@app.get("/desktop")
async def desktop():
    """Serve the ambient desktop layer template."""
    template_path = TEMPLATE_DIR / "template-desktop.html"
    if not template_path.exists():
        template_path = TEMPLATE_DIR / "template-live.html"
    html = template_path.read_text()
    libs_dir = TEMPLATE_DIR / ".lib_cache"
    for lib_name, lib_file in [("PLOTLY_SCRIPT", "plotly-2.27.0.min.js"),
                                ("THREEJS_SCRIPT", "three-0.160.0.min.js"),
                                ("FORCEGRAPH_SCRIPT", "3d-force-graph.min.js")]:
        pf = libs_dir / lib_file
        if pf.exists():
            html = html.replace(f"__{lib_name}__", f"<script>{pf.read_text()}</script>")
    html = html.replace("__DATA_JSON__", json.dumps(current_data))
    return HTMLResponse(content=html)



@app.get("/api/graph")
async def api_graph():
    return current_data


@app.get("/api/stats")
async def api_stats():
    return current_data.get("stats", {})


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type": "initial",
            "stats": current_data["stats"],
            "nodes": current_data["nodes"],
            "edges": current_data["edges"],
            "categories": current_data["categories"],
            "weights": current_data["weights"],
        })
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


def main():
    global current_data, current_hash, reload_fn

    parser = argparse.ArgumentParser(description="Neural Memory Live Dashboard Server")
    parser.add_argument("--db", default=None, help="Force SQLite database path")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-tls", action="store_true")
    parser.add_argument("--desktop-layer", action="store_true")
    args = parser.parse_args()

    if args.desktop_layer:
        logging.getLogger().setLevel(logging.WARNING)

    current_data, reload_fn = load_data(args)
    current_hash = compute_data_hash(current_data)

    cert_dir = TEMPLATE_DIR / ".certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = str(cert_dir / "dashboard.crt")
    key_file = str(cert_dir / "dashboard.key")

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

    protocol = "https" if not args.no_tls else "http"
    src = current_data["stats"]["source"]
    print(f"\n{'='*60}")
    print(f"  Neural Memory — LIVE Dashboard ({src})")
    print(f"  {protocol}://localhost:{args.port}/")
    print(f"  WebSocket: {protocol.replace('http','ws')}://localhost:{args.port}/ws/stream")
    print(f"  {current_data['stats']['memories']} memories, {current_data['stats']['connections']} connections")
    print(f"  Polling every {POLL_INTERVAL}s for changes")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info" if not args.desktop_layer else "warning",
                **ssl_kwargs)


if __name__ == "__main__":
    main()
