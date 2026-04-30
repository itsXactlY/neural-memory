#!/usr/bin/env python3
"""
generate.py - Generate interactive Mazemaker dashboard HTML.

Reads from SQLite and produces a self-contained interactive HTML file
with 3D force graph + Plotly visualizations.

Usage:
    python generate.py                          # SQLite, output ~/neural_memory_dashboard.html
    python generate.py --db /path/to/memory.db  # Custom SQLite path
    python generate.py -o /tmp/dashboard.html   # Custom output path
    python generate.py --serve                  # Generate + serve via HTTPS (auto-cert)
    python generate.py --serve --port 8443      # Custom port
"""
import argparse
import json
import os
import sqlite3
import sys
import hashlib
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import tempfile
import subprocess

# Embedding dim is read dynamically from DB at runtime (not hardcoded)
DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
TEMPLATE_PATH = Path(__file__).parent / "template.html"
LIB_CACHE = Path(__file__).parent / ".lib_cache"

# Library versions and URLs
PLOTLY_VERSION = "2.27.0"
PLOTLY_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"
THREEJS_VERSION = "0.160.0"
THREEJS_URL = f"https://cdn.jsdelivr.net/npm/three@{THREEJS_VERSION}/build/three.min.js"
FORCEGRAPH_URL = "https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"


def _download_js(url: str, cache_file: Path, label: str, min_size: int = 100_000) -> str:
    """Download a JS library to local cache. Returns JS content or None on failure."""
    if cache_file.exists() and cache_file.stat().st_size > min_size:
        return cache_file.read_text()

    LIB_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} (one-time)...")

    import urllib.request
    js = None

    # Try urllib first
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "mazemaker-dashboard/3.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            js = resp.read().decode("utf-8")
    except Exception:
        pass

    # Fallback: curl
    if not js or len(js) < min_size:
        try:
            result = subprocess.run(
                ["curl", "-sSL", "--insecure", url],
                capture_output=True, text=True, timeout=60
            )
            js = result.stdout
        except Exception as e:
            print(f"  WARNING: Could not download {label} ({e}). Will use CDN fallback.")
            return None

    if not js or len(js) < min_size:
        print(f"  WARNING: {label} download too small ({len(js) if js else 0} bytes). CDN fallback.")
        return None

    cache_file.write_text(js)
    print(f"    Cached: {cache_file} ({len(js) // 1024} KB)")
    return js


def ensure_libraries() -> dict:
    """Download all required JS libraries. Returns {name: js_content_or_none}."""
    libs = {}

    plotly_cache = LIB_CACHE / f"plotly-{PLOTLY_VERSION}.min.js"
    libs["plotly"] = _download_js(PLOTLY_URL, plotly_cache, f"Plotly {PLOTLY_VERSION}", 1_000_000)

    threejs_cache = LIB_CACHE / f"three-{THREEJS_VERSION}.min.js"
    libs["threejs"] = _download_js(THREEJS_URL, threejs_cache, f"Three.js {THREEJS_VERSION}", 300_000)

    forcegraph_cache = LIB_CACHE / "3d-force-graph.min.js"
    libs["forcegraph"] = _download_js(FORCEGRAPH_URL, forcegraph_cache, "3d-force-graph", 50_000)

    return libs


def read_sqlite(db_path: str) -> dict:
    """Extract visualization data from SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # All memories with category + degree
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
            "id": r[0],
            "label": label[:50],
            "category": _categorize(label),
            "content_length": len(r[2]) if r[2] else 0,
            "salience": r[3] or 1.0,
            "access_count": r[4] or 0,
            "out_degree": r[5],
            "in_degree": r[6],
            "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })

    # Top nodes for graph (more for 3D — it handles scale better)
    hub_ids = [n["id"] for n in nodes[:120]]
    id_set = set(hub_ids)

    # Connections between hubs
    cur.execute("SELECT source_id, target_id, weight FROM connections")
    edges = []
    for r in cur.fetchall():
        if r[0] in id_set and r[1] in id_set:
            edges.append({"source": r[0], "target": r[1], "weight": round(r[2], 4)})

    # Category distribution
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

    # Weight distribution
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

    # Stats
    cur.execute("SELECT COUNT(*) FROM memories")
    n_mem = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM connections")
    n_conn = cur.fetchone()[0]
    # Read actual embedding dim from DB
    cur.execute("SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1")
    emb_row = cur.fetchone()
    actual_dim = (emb_row[0] // 4) if emb_row else 1024

    conn.close()

    return {
        "nodes": nodes,
        "edges": edges,
        "categories": categories,
        "weights": weights,
        "stats": {
            "memories": n_mem,
            "connections": n_conn,
            "embedding_dim": actual_dim,
            "source": "SQLite",
            "path": db_path,
        },
    }


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


def generate_self_signed_cert(cert_path: str, key_path: str):
    """Generate a self-signed TLS certificate using openssl."""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        import time
        age_days = (time.time() - os.path.getmtime(cert_path)) / 86400
        if age_days < 360:
            return

    print("Generating self-signed TLS certificate...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "ec",
        "-pkeyopt", "ec_paramgen_curve:prime256v1",
        "-keyout", key_path, "-out", cert_path,
        "-days", "3650", "-nodes",
        "-subj", "/CN=mazemaker-dashboard/O=Mazemaker/C=DE",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0",
    ], capture_output=True, check=True)
    print(f"  Certificate: {cert_path}")
    print(f"  Key: {key_path}")


def generate_html(data: dict, output_path: str):
    """Read template, embed JS libraries locally, and inject data."""
    template = TEMPLATE_PATH.read_text()

    libs = ensure_libraries()

    # Embed Plotly
    if libs["plotly"]:
        plotly_tag = f"<script>{libs['plotly']}</script>"
    else:
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"></script>'

    # Embed Three.js
    if libs["threejs"]:
        threejs_tag = f"<script>{libs['threejs']}</script>"
    else:
        threejs_tag = f'<script src="https://cdn.jsdelivr.net/npm/three@{THREEJS_VERSION}/build/three.min.js"></script>'

    # Embed 3d-force-graph
    if libs["forcegraph"]:
        forcegraph_tag = f"<script>{libs['forcegraph']}</script>"
    else:
        forcegraph_tag = '<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"></script>'

    html = template.replace("__PLOTLY_SCRIPT__", plotly_tag)
    html = html.replace("__THREEJS_SCRIPT__", threejs_tag)
    html = html.replace("__FORCEGRAPH_SCRIPT__", forcegraph_tag)
    html = html.replace("__DATA_JSON__", json.dumps(data))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    size_kb = len(html) // 1024
    print(f"Dashboard: {output_path} ({size_kb} KB)")
    print(f"  {data['stats']['memories']} memories, {data['stats']['connections']} connections")
    print(f"  Source: {data['stats']['source']} ({data['stats']['path']})")
    embedded = [k for k, v in libs.items() if v]
    cdn = [k for k, v in libs.items() if not v]
    print(f"  Embedded: {', '.join(embedded) if embedded else 'none'}")
    if cdn:
        print(f"  CDN fallback: {', '.join(cdn)}")


def serve_https(output_path: str, port: int):
    """Serve the dashboard via HTTPS with auto-generated self-signed cert."""
    cert_dir = Path(__file__).parent / ".certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = str(cert_dir / "dashboard.crt")
    key_file = str(cert_dir / "dashboard.key")

    generate_self_signed_cert(cert_file, key_file)

    os.chdir(os.path.dirname(output_path) or ".")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), QuietHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    print(f"\n{'='*60}")
    print(f"  Mazemaker Dashboard — HTTPS")
    print(f"  https://localhost:{port}/{os.path.basename(output_path)}")
    print(f"  https://<your-ip>:{port}/{os.path.basename(output_path)}")
    print(f"")
    print(f"  Certificate: auto-generated self-signed (EC P-256)")
    print(f"  Browser will show 'Not Secure' — click Advanced → Proceed")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Generate Mazemaker Dashboard")
    parser.add_argument("--db", default=DEFAULT_SQLITE, help="SQLite database path")
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/neural_memory_dashboard.html"))
    parser.add_argument("--serve", action="store_true", help="Serve via HTTPS after generating")
    parser.add_argument("--port", type=int, default=8443, help="HTTPS port (default: 8443)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)
    data = read_sqlite(args.db)

    generate_html(data, args.output)

    if args.serve:
        serve_https(args.output, args.port)


if __name__ == "__main__":
    main()
