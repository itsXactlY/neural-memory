#!/usr/bin/env python3.11
"""nm_digest.py — single-command status snapshot of neural-memory + bridge + repo.

Built for cross-agent visibility. Anyone (NM lane / AE-builder lane / hermes /
tito at terminal) can run:
    python3 tools/nm_digest.py

and get a 1-page summary of:
- Live DB state (counts, top entities, recent saves)
- Bridge inbox state (open messages, threads)
- Branch state (commits ahead, last commit, push status)
- Recent ingest summary
- Active processes

Designed for end-of-session wrap or mid-session "where do we stand?" checks.
Read-only — never writes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_NM_DB = Path.home() / ".neural_memory" / "memory.db"
_BRIDGE_AGENTS = Path.home() / ".hermes" / "agent-bridge" / "agents"
_HMS = lambda secs: f"{secs/3600:.1f}h" if secs > 3600 else f"{secs/60:.0f}m" if secs > 60 else f"{secs:.0f}s"


def _section(title: str) -> str:
    return f"\n=== {title} ===\n"


def _git(args: list[str], cwd: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd or str(_ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception as e:
        return f"(git err: {e})"


def db_section() -> str:
    if not _NM_DB.exists():
        return f"DB not found at {_NM_DB}"
    conn = sqlite3.connect(str(_NM_DB))
    lines = [f"DB: {_NM_DB}"]
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    entities = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind='entity'"
    ).fetchone()[0]
    lines.append(f"  memories: {total}  edges: {edges}  entities: {entities}")

    by_kind = dict(conn.execute(
        "SELECT COALESCE(kind,'<NULL>'), COUNT(*) FROM memories GROUP BY kind"
    ).fetchall())
    lines.append(f"  by kind: {by_kind}")

    top_ents = conn.execute(
        "SELECT label, JSON_EXTRACT(metadata_json, '$.frequency') AS freq "
        "FROM memories WHERE kind='entity' "
        "ORDER BY freq DESC LIMIT 10"
    ).fetchall()
    lines.append("  top 10 entities by mention frequency:")
    for label, freq in top_ents:
        lines.append(f"    {(freq or 0):>4}  {label}")

    # Recent saves (kind != entity) by created_at
    recent = conn.execute(
        "SELECT id, label, kind, datetime(created_at,'unixepoch') AS ts "
        "FROM memories WHERE kind IS NULL OR kind != 'entity' "
        "ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    lines.append("  most-recent 5 user memories:")
    for mid, label, kind, ts in recent:
        lines.append(f"    {ts}  id={mid:5d}  [{(kind or '?'):12s}]  {(label or '')[:50]}")

    fts = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='memories_fts'"
    ).fetchone()
    if fts:
        n_fts = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        n_non_entity = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind IS NULL OR kind != 'entity'"
        ).fetchone()[0]
        delta = n_fts - n_non_entity
        lines.append(f"  FTS5 sync delta: {delta} (0 = clean)")

    conn.close()
    return "\n".join(lines)


def bridge_section() -> str:
    lines = []
    if not _BRIDGE_AGENTS.exists():
        return f"  bridge dir not found: {_BRIDGE_AGENTS}"
    for agent_dir in sorted(_BRIDGE_AGENTS.iterdir()):
        if not agent_dir.is_dir():
            continue
        inbox = agent_dir / "inbox.jsonl"
        outbox = agent_dir / "outbox.jsonl"
        if not inbox.exists() and not outbox.exists():
            continue
        in_count = 0
        unack_count = 0
        if inbox.exists():
            for line in inbox.read_text(errors="ignore").splitlines():
                if line.strip():
                    in_count += 1
                    try:
                        msg = json.loads(line)
                        if not msg.get("acknowledged"):
                            unack_count += 1
                    except Exception:
                        pass
        out_count = 0
        if outbox.exists():
            out_count = sum(1 for L in outbox.read_text(errors="ignore").splitlines() if L.strip())
        lines.append(f"  {agent_dir.name:25s}  in={in_count:4d}  out={out_count:4d}  (unack ~{unack_count})")
    return "\n".join(lines) if lines else "  (no agents)"


def repo_section() -> str:
    lines = []
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["rev-parse", "--short", "HEAD"])
    upstream = _git(["rev-parse", "--abbrev-ref", "@{u}"])
    ahead_behind = _git(["rev-list", "--left-right", "--count", "HEAD...@{u}"])
    last_commit = _git(["log", "-1", "--pretty=format:%h %ar  %s"])

    lines.append(f"  branch: {branch}  HEAD: {head}")
    lines.append(f"  upstream: {upstream}")
    if ahead_behind:
        try:
            ahead, behind = ahead_behind.split()
            lines.append(f"  ahead: {ahead}  behind: {behind}")
        except ValueError:
            lines.append(f"  rev-list output: {ahead_behind}")
    lines.append(f"  last commit: {last_commit}")

    # Last 5 commits
    recent = _git(["log", "-5", "--pretty=format:    %h %ar  %s"])
    lines.append("  recent 5:")
    lines.append(recent)
    return "\n".join(lines)


def proc_section() -> str:
    lines = []
    try:
        out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5,
        )
        rows = out.stdout.splitlines()
    except Exception:
        return "  (ps unavailable)"
    interesting = []
    for r in rows:
        if any(kw in r for kw in (
            "lme_real.py", "ingest_ae_corpus.py", "hermes_cli", "wa_hermes_adapter",
            "neural-memory", "phase7_audit", "post_ingest_sanity", "lme_eval",
        )):
            if "grep" in r or "ps aux" in r:
                continue
            cols = r.split()
            if len(cols) >= 11:
                pid = cols[1]
                cpu = cols[2]
                mem = cols[3]
                cmd = " ".join(cols[10:])[:80]
                interesting.append(f"  pid={pid} cpu={cpu}% mem={mem}%  {cmd}")
    return "\n".join(interesting[:8]) if interesting else "  (none of interest running)"


def scoring_weights_section() -> str:
    """Surface DEFAULT_WEIGHTS so operators can see channel balance."""
    try:
        sys.path.insert(0, str(_ROOT / "python"))
        from scoring import DEFAULT_WEIGHTS
        lines = ["  channel       weight"]
        for ch in ("semantic", "sparse", "graph", "temporal",
                   "entity", "procedural", "locus", "rrf"):
            w = DEFAULT_WEIGHTS.get(ch, 0.0)
            lines.append(f"  {ch:13s} {w:.2f}")
        total = sum(DEFAULT_WEIGHTS.values())
        lines.append(f"  {'TOTAL':13s} {total:.2f}")
        return "\n".join(lines)
    except Exception as e:
        return f"  (could not load DEFAULT_WEIGHTS: {e})"


def phase7_5_wiring_section() -> str:
    """One-glance Phase 7.5 wiring scoreboard."""
    if not _NM_DB.exists():
        return "  (DB not found)"
    conn = sqlite3.connect(str(_NM_DB))
    try:
        proc_n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE procedural_score IS NOT NULL"
        ).fetchone()[0]
        ent_n = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE edge_type='mentions_entity'"
        ).fetchone()[0]
        contradict_n = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE edge_type='contradicts'"
        ).fetchone()[0]
        insight_n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind='dream_insight'"
        ).fetchone()[0]
        locus_n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE kind='locus'"
        ).fetchone()[0]
    finally:
        conn.close()
    lines = [
        "  Subphase  Status     Live signal",
        f"  α  proc   SHIPPED   procedural_score populated rows: {proc_n}",
        f"  β  entity SHIPPED   mentions_entity edges:           {ent_n}",
        f"  γ  stale  SHIPPED   computed at-query from age (no row count)",
        f"  δ  contra SHIPPED   contradicts edges:               {contradict_n}",
        f"  ε  locus  SHIPPED   locus nodes:                     {locus_n}",
        f"  D5 (Valiendo lane):  dream_insight nodes:            {insight_n}",
    ]
    return "\n".join(lines)


def ingest_section() -> str:
    """Look for traces of recent ingest_ae_corpus.py runs in the live DB metadata."""
    if not _NM_DB.exists():
        return "  (DB not found)"
    conn = sqlite3.connect(str(_NM_DB))
    rows = conn.execute(
        "SELECT JSON_EXTRACT(metadata_json,'$.source_label') AS src, COUNT(*) AS n, "
        "       datetime(MAX(transaction_time),'unixepoch') AS last_ingest "
        "FROM memories "
        "WHERE JSON_EXTRACT(metadata_json,'$.ingested_by')='ingest_ae_corpus.py' "
        "GROUP BY src ORDER BY n DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return "  (no ingest traces found)"
    lines = ["  source_label              count  last_ingest"]
    for src, n, last_ts in rows:
        lines.append(f"  {src:25s}  {n:5d}  {last_ts or '?'}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true",
                    help="Output structured JSON (machine-readable)")
    args = p.parse_args()

    if args.json:
        # Minimal structured for scripting
        out_data = {
            "db_path": str(_NM_DB),
            "db_exists": _NM_DB.exists(),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _git(["rev-parse", "--short", "HEAD"]),
            "ahead_behind": _git(["rev-list", "--left-right", "--count", "HEAD...@{u}"]),
        }
        if _NM_DB.exists():
            conn = sqlite3.connect(str(_NM_DB))
            out_data["memories"] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            out_data["edges"] = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
            out_data["entities"] = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE kind='entity'"
            ).fetchone()[0]
            conn.close()
        print(json.dumps(out_data, indent=2))
        return 0

    print("nm_digest — neural-memory + bridge + repo status snapshot")
    print(_section("Repo state"))
    print(repo_section())
    print(_section("Live DB"))
    print(db_section())
    print(_section("Scoring weights (DEFAULT_WEIGHTS)"))
    print(scoring_weights_section())
    print(_section("Phase 7.5 wiring"))
    print(phase7_5_wiring_section())
    print(_section("Ingest sources"))
    print(ingest_section())
    print(_section("Bridge mailboxes"))
    print(bridge_section())
    print(_section("Active processes"))
    print(proc_section())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
