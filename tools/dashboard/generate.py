#!/usr/bin/env python3
"""
generate.py - Generate interactive Neural Memory dashboard HTML.

Reads from SQLite (default) or MSSQL and produces a self-contained
interactive HTML file with Plotly visualizations.

Usage:
    python generate.py                          # SQLite, output ~/neural_memory_dashboard.html
    python generate.py --mssql                  # MSSQL (NeuralMemory DB)
    python generate.py --db /path/to/memory.db  # Custom SQLite path
    python generate.py -o /tmp/dashboard.html   # Custom output path
"""
import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

EMBEDDING_DIM = 384
DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
TEMPLATE_PATH = Path(__file__).parent / "template.html"


def read_sqlite(db_path: str) -> dict:
    """Extract visualization data from SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Memories with category + degree
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
            "label": label[:40],
            "category": _categorize(label),
            "content_length": len(r[2]) if r[2] else 0,
            "salience": r[3] or 1.0,
            "access_count": r[4] or 0,
            "out_degree": r[5],
            "in_degree": r[6],
            "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })

    # Top 50 hub nodes for network graph
    hub_ids = [n["id"] for n in nodes[:50]]
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

    conn.close()

    return {
        "nodes": nodes,
        "edges": edges,
        "categories": categories,
        "weights": weights,
        "stats": {
            "memories": n_mem,
            "connections": n_conn,
            "embedding_dim": EMBEDDING_DIM,
            "source": "SQLite",
            "path": db_path,
        },
    }


def read_mssql(server="localhost", database="NeuralMemory",
               username="SA", password=None) -> dict:
    """Extract visualization data from MSSQL."""
    try:
        import pyodbc
    except ImportError:
        print("pyodbc required for MSSQL. pip install pyodbc")
        sys.exit(1)

    if not password:
        print("MSSQL password required. Use --mssql-password or set MSSQL_PASSWORD env var.")
        sys.exit(1)

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={username};PWD={password};"
        f"TrustServerCertificate=yes;"
    )
    conn = pyodbc.connect(conn_str, autocommit=True)
    cur = conn.cursor()

    # Top nodes by degree
    cur.execute("""
        SELECT TOP 200 m.id, m.label, LEN(ISNULL(m.content,'')) AS clen,
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
            "id": r[0], "label": label[:40], "category": _categorize(label),
            "content_length": r[2] or 0, "salience": r[3] or 1.0,
            "access_count": r[4] or 0, "out_degree": r[5],
            "in_degree": r[6], "total_degree": r[5] + r[6],
            "avg_weight": round(r[7], 4),
        })

    hub_ids = [n["id"] for n in nodes[:50]]
    id_set = set(hub_ids)

    id_list = ",".join(str(x) for x in hub_ids)
    cur.execute(f"SELECT source_id, target_id, weight FROM connections WHERE source_id IN ({id_list}) AND target_id IN ({id_list})")
    edges = [{"source": r[0], "target": r[1], "weight": round(r[2], 4)} for r in cur.fetchall()]

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

    conn.close()

    return {
        "nodes": nodes, "edges": edges,
        "categories": categories, "weights": weights,
        "stats": {"memories": n_mem, "connections": n_conn,
                  "embedding_dim": EMBEDDING_DIM, "source": "MSSQL", "path": f"{server}/{database}"},
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


def generate_html(data: dict, output_path: str):
    """Read template and inject data."""
    template = TEMPLATE_PATH.read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    size_kb = len(html) // 1024
    print(f"Dashboard: {output_path} ({size_kb} KB)")
    print(f"  {data['stats']['memories']} memories, {data['stats']['connections']} connections")
    print(f"  Source: {data['stats']['source']} ({data['stats']['path']})")


def main():
    parser = argparse.ArgumentParser(description="Generate Neural Memory Dashboard")
    parser.add_argument("--db", default=DEFAULT_SQLITE, help="SQLite database path")
    parser.add_argument("--mssql", action="store_true", help="Use MSSQL instead of SQLite")
    parser.add_argument("--mssql-server", default="localhost")
    parser.add_argument("--mssql-database", default="NeuralMemory")
    parser.add_argument("--mssql-username", default="SA")
    parser.add_argument("--mssql-password", default=None, help="Or set MSSQL_PASSWORD env var")
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/neural_memory_dashboard.html"))
    args = parser.parse_args()

    if args.mssql:
        pw = args.mssql_password or os.environ.get("MSSQL_PASSWORD")
        data = read_mssql(args.mssql_server, args.mssql_database, args.mssql_username, pw)
    else:
        if not os.path.exists(args.db):
            print(f"Database not found: {args.db}")
            sys.exit(1)
        data = read_sqlite(args.db)

    generate_html(data, args.output)


if __name__ == "__main__":
    main()
