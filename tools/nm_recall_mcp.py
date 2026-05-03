#!/usr/bin/env python3.11
"""nm_recall_mcp.py — Model Context Protocol stdio server for cross-agent
neural-memory access.

Wraps `memory_client.NeuralMemory` over JSON-RPC so any MCP-compatible
agent (Claude Code, Codex, Hermes) can query the same substrate.

Pattern matches `claude-hermes-bridge/scripts/agent_bridge_mcp.mjs` —
Node.js precedent — but implemented in Python because memory_client
imports directly.

Tools exposed:
  - nm_recall: hybrid_recall with bench-validated config (current AE-domain
    R@5 reported live by ``_bench_authority()`` from the most recent
    ``~/.neural_memory/bench-history/ae-domain-*.json`` artifact)
  - nm_sparse_search: BM25 sparse-only (cheaper, no rerank load)
  - nm_remember: write a new memory
  - nm_status: substrate health snapshot (counts, sources)
  - nm_audit: per-source row counts + recent ingest summary

Cross-agent design:
  - Single source of truth: ~/.neural_memory/memory.db (shared via FS)
  - SQLite WAL mode is multi-reader/writer-safe
  - Doesn't conflict with the Hermes plugin (both can run; same DB)
  - Stateless server: starts fresh on each MCP-client connection

Install (Claude Code):
  Add to ~/.claude.json mcpServers:
    "nm-recall": {
      "command": "/Users/tito/lWORKSPACEl/research/neural-memory/tools/nm_recall_mcp.py"
    }

Install (Codex / Hermes): same command, their MCP config schema.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

# Lazy-import to avoid heavy embedder load at startup
_mem = None

# Default location of bench-history artifacts produced by the F9 AE-domain
# bench harness. Overridable for tests via _bench_authority(bench_dir=...).
_DEFAULT_BENCH_DIR = Path.home() / ".neural_memory" / "bench-history"


def _bench_authority(bench_dir: Path | None = None) -> tuple[str, str]:
    """Return current AE-domain bench R@5 from the most recent artifact.

    Reads the latest ``ae-domain-*.json`` under ``bench_dir`` (default
    ``~/.neural_memory/bench-history/``) and extracts the top-level
    ``global_r@5`` field per the F9 artifact format.

    Returns a ``(r_at_5, artifact_name)`` tuple of strings. Both values
    fall back to ``"unknown"`` if no artifact is present, the file cannot
    be read, the JSON is malformed, or the expected key is missing.
    Stringly-typed by design so the value can be safely interpolated
    into prose without per-call float-format guards.
    """
    bench_dir = bench_dir or _DEFAULT_BENCH_DIR
    try:
        artifacts = sorted(bench_dir.glob("ae-domain-*.json"))
        if not artifacts:
            return ("unknown", "unknown")
        latest = artifacts[-1]
        with latest.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        val = data.get("global_r@5")
        if val is None:
            return ("unknown", latest.name)
        return (f"{float(val):.4f}", latest.name)
    except Exception:
        return ("unknown", "unknown")


def _get_mem():
    global _mem
    if _mem is None:
        from memory_client import NeuralMemory
        # Reviewer-round-6 fix 2026-05-02: use_hnsw=True is required to keep
        # dense-channel latency under control. Without HNSW, every dense
        # channel call linear-scans 12k+ memories per query, blowing up p50.
        _mem = NeuralMemory(
            embedding_backend="auto",
            use_cpp=False,
            use_hnsw=True,
        )
    return _mem


# ---- MCP JSON-RPC protocol ------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "nm-recall"
SERVER_VERSION = "0.1.0"


_R5, _R5_ARTIFACT = _bench_authority()

TOOLS = [
    {
        "name": "nm_recall",
        "description": (
            "Multi-channel hybrid retrieval over the AE neural-memory "
            "substrate. Bench-validated config: dense + sparse + graph + "
            "temporal channels with cross-encoder rerank (auto-skip for "
            "Spanish queries to avoid English-rerank regression). "
            "Returns top-K memories with combined score, channel "
            f"breakdown, and per-feature trace. Current bench R@5: {_R5} "
            f"(from {_R5_ARTIFACT})."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "free-text query"},
                "k": {"type": "integer", "default": 5},
                "kind": {
                    "type": "string",
                    "description": "filter by kind: experience, procedural, world, dream_insight, locus",
                },
                "as_of": {
                    "type": "number",
                    "description": "bi-temporal cutoff (unix ts)",
                },
                "rerank": {
                    "type": "boolean",
                    "default": True,
                    "description": "cross-encoder rerank (auto-skipped for Spanish)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nm_sparse_search",
        "description": (
            "FTS5 BM25 sparse-only retrieval. Cheaper than nm_recall — "
            "no model load, no rerank. Best for explicit-keyword queries "
            "or when caller doesn't need semantic similarity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "nm_remember",
        "description": (
            "Write a new memory to the substrate. Auto-classifies kind "
            "if not provided. Auto-extracts entities. Returns memory id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "label": {"type": "string", "default": ""},
                "kind": {
                    "type": "string",
                    "description": "experience | procedural | world | locus | profile_trait | etc.",
                },
                "source": {
                    "type": "string",
                    "description": "provenance label (e.g. 'whatsapp', 'cli', 'mcp_caller')",
                },
                "salience": {
                    "type": "number",
                    "default": 1.0,
                    "description": "memory weight 0-2",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "nm_status",
        "description": (
            "Substrate health snapshot: total memories, edges, top "
            "entities, FTS5 sync delta. Read-only."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nm_audit",
        "description": (
            "Per-source ingest breakdown + recent-saves sample. Read-only. "
            "Useful for understanding what's in the substrate."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _ok(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _content(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


# ---- Tool implementations -------------------------------------------------

def _tool_nm_recall(args: dict) -> dict:
    query = args["query"]
    k = int(args.get("k", 5))
    kind = args.get("kind")
    as_of = args.get("as_of")
    rerank = args.get("rerank", True)
    mem = _get_mem()
    results = mem.hybrid_recall(
        query, k=k, kind=kind, as_of=as_of, rerank=bool(rerank),
    )
    # Strip embedding blobs to keep JSON payload reasonable
    cleaned = []
    for r in results:
        cleaned.append({
            "id": r.get("id"),
            "label": r.get("label"),
            "content": (r.get("content") or "")[:1500],
            "kind": r.get("kind"),
            "combined": r.get("combined"),
            "channels": r.get("channels"),
            "_trace": r.get("_trace"),
        })
    return _content(json.dumps({"query": query, "k": k, "results": cleaned}, indent=2))


def _tool_nm_sparse_search(args: dict) -> dict:
    query = args["query"]
    k = int(args.get("k", 5))
    mem = _get_mem()
    results = mem.sparse_search(query, k=k)
    cleaned = [
        {"id": r.get("id"), "label": r.get("label"),
         "content": (r.get("content") or "")[:800]}
        for r in results
    ]
    return _content(json.dumps({"query": query, "k": k, "results": cleaned}, indent=2))


def _tool_nm_remember(args: dict) -> dict:
    text = args["text"]
    label = args.get("label", "")
    kind = args.get("kind")
    source = args.get("source", "mcp_caller")
    salience = float(args.get("salience", 1.0))
    mem = _get_mem()
    mid = mem.remember(
        text, label=label, kind=kind, source=source, salience=salience,
    )
    return _content(json.dumps({"id": mid, "saved": True}))


def _tool_nm_status(args: dict) -> dict:
    import sqlite3
    db_path = Path.home() / ".neural_memory" / "memory.db"
    # Reviewer-round-6 fix 2026-05-02: check_same_thread=False for future
    # async-transport MCP clients (stdio loop is single-threaded today).
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        entities = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind='entity'"
        ).fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        non_entity = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind IS NULL OR kind != 'entity'"
        ).fetchone()[0]
        top_entities = conn.execute(
            "SELECT label, JSON_EXTRACT(metadata_json,'$.frequency') AS f "
            "FROM memories WHERE kind='entity' ORDER BY f DESC LIMIT 8"
        ).fetchall()
    finally:
        conn.close()
    # Reviewer-round-6 fix 2026-05-02: report fts5 + non_entity counts
    # as separate fields. Previous fts5_sync_delta = fts - non_entity
    # could mislead because some entity rows occasionally end up in FTS5
    # (per test_schema_upgrade.py:219). Caller can compute drift as
    # they prefer.
    return _content(json.dumps({
        "db_path": str(db_path),
        "total_memories": total,
        "entities": entities,
        "edges": edges,
        "memories_fts_count": fts,
        "non_entity_memories_count": non_entity,
        "top_entities_by_freq": [{"label": l, "freq": f} for l, f in top_entities],
    }, indent=2))


def _tool_nm_audit(args: dict) -> dict:
    import sqlite3
    db_path = Path.home() / ".neural_memory" / "memory.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        sources = conn.execute(
            "SELECT JSON_EXTRACT(metadata_json,'$.source_label') AS src, COUNT(*) "
            "FROM memories WHERE JSON_EXTRACT(metadata_json,'$.ingested_by')='ingest_ae_corpus.py' "
            "GROUP BY src ORDER BY 2 DESC LIMIT 15"
        ).fetchall()
        recent = conn.execute(
            "SELECT id, label, datetime(transaction_time,'unixepoch') "
            "FROM memories WHERE kind IS NULL OR kind != 'entity' "
            "ORDER BY transaction_time DESC LIMIT 5"
        ).fetchall()
    finally:
        conn.close()
    return _content(json.dumps({
        "ingest_sources": [{"source_label": s, "count": n} for s, n in sources],
        "recent_5_writes": [
            {"id": i, "label": l, "ts": t} for i, l, t in recent
        ],
    }, indent=2))


_TOOL_HANDLERS = {
    "nm_recall": _tool_nm_recall,
    "nm_sparse_search": _tool_nm_sparse_search,
    "nm_remember": _tool_nm_remember,
    "nm_status": _tool_nm_status,
    "nm_audit": _tool_nm_audit,
}


# ---- Main loop ------------------------------------------------------------

def _handle(req: dict) -> dict | None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {}},
        })

    if method == "notifications/initialized":
        return None  # notification, no response

    if method == "tools/list":
        return _ok(req_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = _TOOL_HANDLERS.get(name)
        if not handler:
            return _err(req_id, -32601, f"unknown tool: {name}")
        try:
            result = handler(args)
            return _ok(req_id, result)
        except Exception as e:
            return _err(req_id, -32603, f"{type(e).__name__}: {e}")

    if method == "ping":
        return _ok(req_id, {})

    return _err(req_id, -32601, f"unknown method: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
