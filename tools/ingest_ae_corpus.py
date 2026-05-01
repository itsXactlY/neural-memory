#!/usr/bin/env python3
"""ingest_ae_corpus.py — seed neural-memory with AE's canonical knowledge.

Walks the AE project's canonical sources (operator skill, LangGraph
context kernel, claude-memory project/feedback docs) and ingests them
as typed memories via NeuralMemory.remember(). Each chunk gets:
    - kind: classified or specified per source category
    - source: 'ae_operator_skill' / 'langgraph_kernel' / 'claude_memory'
    - origin_system: 'ae'
    - valid_from: file mtime
    - metadata: { source_path, section_heading, file_mtime }
    - salience: 0.7 (medium-low so live hermes saves stay primary)

Idempotent: dedupes by content hash. Re-running skips already-ingested
content.

Usage:
    python3 tools/ingest_ae_corpus.py --dry-run     # show what would ingest
    python3 tools/ingest_ae_corpus.py               # actually ingest
    python3 tools/ingest_ae_corpus.py --db PATH     # override DB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

# Canonical source roots, in priority order.
_AE_OPERATOR_SKILL = Path("/Users/tito/.hermes/skills/angels-electric/AE_OPERATOR_SKILL.md")
_AE_SKILL_DIR = Path("/Users/tito/.hermes/skills/angels-electric")  # SKILL.md + references/
_LANGGRAPH_KERNEL = Path("/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/context-kernel")
_CLAUDE_MEMORY = Path("/Users/tito/.claude/projects/-Users-tito/memory")
_VALIENDO_HANDOFFS = Path("/Users/tito/.hermes/artifacts/valiendo/handoffs")
_BRIDGE_AGENTS_DIR = Path("/Users/tito/.hermes/agent-bridge/agents")  # bridge inbox.jsonl per agent

# Bridge messages: limit to last N days to avoid pulling stale history
_BRIDGE_MAX_AGE_DAYS = 60

# Memory-file kind heuristics (file-name prefix → memory kind).
_MEMORY_PREFIX_TO_KIND = {
    "feedback_": "procedural",      # behavioral rules
    "user_":     "profile_trait",   # user profile traits
    "project_":  "experience",      # project state
    "reference_": "world",          # canonical reference docs
}

# Chunking limits.
_MAX_CHUNK_CHARS = 4000     # ~1000 tokens; embedder-friendly
_MIN_CHUNK_CHARS = 50       # skip whitespace-only fragments

# Skip files larger than this (avoid huge addendum dumps that overflow context)
_MAX_FILE_BYTES = 200_000

# Skip these file-name patterns (transient / huge / not durable knowledge)
_SKIP_PATTERNS = [
    re.compile(r"handoff_neural_memory_lane_"),  # one-shot session handoff
    re.compile(r"reference_neural_memory_execution_addendum"),  # 1800-line spec
    re.compile(r"reference_neural_memory_unified_integration_handoff"),  # 2200-line spec
    re.compile(r"reference_sprint2_phase7_kickoff_prompt"),  # session-specific
    re.compile(r"^MEMORY\.md$"),  # index, not content
]


def _chunk_markdown(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split markdown text into (section_heading, body) chunks. Splits
    on ## headers; bodies bigger than max_chars are further split on \n\n."""
    if not text:
        return []
    sections: list[tuple[str, str]] = []
    parts = re.split(r"^(##\s+.+)$", text, flags=re.MULTILINE)
    # parts: [pre-header-text, header1, body1, header2, body2, ...]
    if parts and parts[0].strip():
        sections.append(("(intro)", parts[0].strip()))
    i = 1
    while i < len(parts):
        heading = parts[i].lstrip("#").strip() if i < len(parts) else ""
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append((heading, body))
        i += 2

    # Now further-split any big bodies
    out: list[tuple[str, str]] = []
    for heading, body in sections:
        if len(body) <= max_chars:
            if len(body) >= _MIN_CHUNK_CHARS:
                out.append((heading, body))
            continue
        # Split on paragraph boundaries
        paragraphs = re.split(r"\n\n+", body)
        buf: list[str] = []
        buf_len = 0
        idx = 0
        for p in paragraphs:
            if buf_len + len(p) > max_chars and buf:
                out.append((f"{heading} [pt {idx + 1}]", "\n\n".join(buf)))
                buf = [p]
                buf_len = len(p)
                idx += 1
            else:
                buf.append(p)
                buf_len += len(p)
        if buf:
            suffix = f" [pt {idx + 1}]" if idx > 0 else ""
            out.append((f"{heading}{suffix}", "\n\n".join(buf)))
    return out


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _kind_for_memory_file(name: str) -> str:
    for prefix, kind in _MEMORY_PREFIX_TO_KIND.items():
        if name.startswith(prefix):
            return kind
    return "world"


def _gather_sources() -> list[dict]:
    """Return list of source descriptors:
    {path, source_label, default_kind, origin_system}"""
    sources: list[dict] = []

    # AE_OPERATOR_SKILL — canonical, treat as procedural by default (it's a skill doc)
    if _AE_OPERATOR_SKILL.exists():
        sources.append({
            "path": _AE_OPERATOR_SKILL,
            "source_label": "ae_operator_skill",
            "default_kind": "procedural",
            "origin_system": "ae",
        })

    # Other docs in ae skill directory (SKILL.md + references/)
    if _AE_SKILL_DIR.exists():
        for p in sorted(_AE_SKILL_DIR.rglob("*.md")):
            if p == _AE_OPERATOR_SKILL:
                continue  # already added
            sources.append({
                "path": p,
                "source_label": "ae_skill_other",
                "default_kind": "procedural",
                "origin_system": "ae",
            })

    # Valiendo handoffs (durable transition records)
    if _VALIENDO_HANDOFFS.exists():
        for p in sorted(_VALIENDO_HANDOFFS.glob("*.md")):
            sources.append({
                "path": p,
                "source_label": "valiendo_handoffs",
                "default_kind": "experience",  # transition events
                "origin_system": "hermes",
            })

    # LangGraph context-kernel docs (architecture / canonical references)
    if _LANGGRAPH_KERNEL.exists():
        for p in sorted(_LANGGRAPH_KERNEL.glob("*.md")):
            sources.append({
                "path": p,
                "source_label": "langgraph_kernel",
                "default_kind": "world",  # canonical references
                "origin_system": "ae",
            })

    # claude-memory durable docs (project state, feedback rules)
    if _CLAUDE_MEMORY.exists():
        for p in sorted(_CLAUDE_MEMORY.glob("*.md")):
            name = p.name
            if any(pat.search(name) for pat in _SKIP_PATTERNS):
                continue
            sources.append({
                "path": p,
                "source_label": "claude_memory",
                "default_kind": _kind_for_memory_file(name),
                "origin_system": "claude_memory",
            })

    return sources


def _existing_content_hashes(store) -> set[str]:
    """Return set of content_hash values already ingested by this script."""
    rows = store.conn.execute(
        "SELECT metadata_json FROM memories WHERE origin_system IN ('ae', 'claude_memory') "
        "AND metadata_json IS NOT NULL"
    ).fetchall()
    hashes: set[str] = set()
    for (meta_json,) in rows:
        try:
            meta = json.loads(meta_json)
            ch = meta.get("content_hash")
            if ch:
                hashes.add(ch)
        except Exception:
            pass
    return hashes


def _gather_bridge_messages() -> list[dict]:
    """Walk bridge agent mailboxes (inbox.jsonl + outbox.jsonl) and return
    one chunk-descriptor per message. Only recent messages (last 60 days)
    to avoid stale archive ingest. Each message is treated as a discrete
    experience-kind memory."""
    if not _BRIDGE_AGENTS_DIR.exists():
        return []

    import datetime as _dt
    cutoff = (_dt.datetime.now(_dt.timezone.utc) -
              _dt.timedelta(days=_BRIDGE_MAX_AGE_DAYS))

    out: list[dict] = []
    seen_msg_ids: set[str] = set()  # avoid double-counting (msg in inbox + outbox)
    for agent_dir in sorted(_BRIDGE_AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        for mailbox_name in ("inbox.jsonl", "outbox.jsonl"):
            mailbox = agent_dir / mailbox_name
            if not mailbox.exists():
                continue
            for line in mailbox.read_text(encoding="utf-8",
                                           errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_id = msg.get("id")
                if not msg_id or msg_id in seen_msg_ids:
                    continue
                # Date filter
                created_at = msg.get("createdAt")
                if created_at:
                    try:
                        ts = _dt.datetime.fromisoformat(
                            created_at.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                        valid_from = ts.timestamp()
                    except Exception:
                        valid_from = mailbox.stat().st_mtime
                else:
                    valid_from = mailbox.stat().st_mtime
                seen_msg_ids.add(msg_id)

                # Build chunk content: subject + body excerpt
                subject = msg.get("subject") or "(no subject)"
                body = msg.get("body") or ""
                # Cap body to avoid pathological-size single-message ingests
                if len(body) > 8000:
                    body = body[:8000] + "\n[...truncated]"
                msg_type = msg.get("type") or "message"
                from_ = msg.get("from") or "?"
                to_ = msg.get("to") or "?"
                heading = f"bridge:{from_}->{to_}: {subject}"[:80]
                content = f"[{from_} → {to_}, {created_at}] {subject}\n\n{body}"
                out.append({
                    "path": mailbox,  # for traceability
                    "source_label": "bridge_mailbox",
                    "default_kind": "experience",
                    "origin_system": "hermes",
                    "_synthetic_chunk": True,  # signal we built our own
                    "_synthetic_heading": heading,
                    "_synthetic_body": content,
                    "_synthetic_mtime": valid_from,
                    "_msg_id": msg_id,
                    "_msg_type": msg_type,
                    "_from": from_,
                    "_to": to_,
                    "_thread_id": msg.get("threadId") or "",
                    "_tags": msg.get("tags") or [],
                })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None, help="DB path override")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would ingest without writing")
    p.add_argument("--limit", type=int, default=None,
                    help="Cap total chunks ingested (for testing)")
    p.add_argument("--salience", type=float, default=0.7,
                    help="Default salience for ingested chunks (0-2)")
    args = p.parse_args()

    sources = _gather_sources()
    print(f"Gathered {len(sources)} canonical source files", file=sys.stderr)

    # Pre-scan: count chunks
    total_chunks = 0
    skipped_oversize = 0
    chunk_inventory: list[dict] = []
    for src in sources:
        path: Path = src["path"]
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size > _MAX_FILE_BYTES:
            skipped_oversize += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        chunks = _chunk_markdown(text)
        for heading, body in chunks:
            chunk_inventory.append({
                "path": path,
                "source_label": src["source_label"],
                "default_kind": src["default_kind"],
                "origin_system": src["origin_system"],
                "heading": heading,
                "body": body,
                "content_hash": _content_hash(body),
                "mtime": path.stat().st_mtime,
            })
            total_chunks += 1

    # Bridge messages — synthetic chunks (one per message, not chunked further)
    bridge_descs = _gather_bridge_messages()
    print(f"Gathered {len(bridge_descs)} bridge messages (last {_BRIDGE_MAX_AGE_DAYS} days)",
          file=sys.stderr)
    for desc in bridge_descs:
        body = desc["_synthetic_body"]
        chunk_inventory.append({
            "path": desc["path"],
            "source_label": desc["source_label"],
            "default_kind": desc["default_kind"],
            "origin_system": desc["origin_system"],
            "heading": desc["_synthetic_heading"],
            "body": body,
            "content_hash": _content_hash(body),
            "mtime": desc["_synthetic_mtime"],
            "_extra_metadata": {
                "msg_id": desc["_msg_id"],
                "msg_type": desc["_msg_type"],
                "from": desc["_from"],
                "to": desc["_to"],
                "thread_id": desc["_thread_id"],
                "tags": desc["_tags"],
            },
        })
        total_chunks += 1

    print(f"Total chunks: {total_chunks}  Skipped oversize: {skipped_oversize}",
          file=sys.stderr)
    if args.limit:
        chunk_inventory = chunk_inventory[: args.limit]
        print(f"Limited to first {args.limit}", file=sys.stderr)

    if args.dry_run:
        print("=== DRY RUN — no writes ===")
        by_source: dict[str, int] = {}
        for c in chunk_inventory:
            by_source[c["source_label"]] = by_source.get(c["source_label"], 0) + 1
        for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src:25s} {n:5d} chunks")
        # Sample first 5
        print("--- sample first 5 ---")
        for c in chunk_inventory[:5]:
            preview = c["body"][:80].replace("\n", " ")
            print(f"  [{c['source_label']:20s}] [{c['default_kind']:10s}] "
                  f"{c['heading'][:30]:30s}  {preview}")
        return 0

    # Real ingest
    import contextlib
    import io
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        from memory_client import NeuralMemory
        kwargs = {"embedding_backend": "auto", "use_cpp": False, "use_hnsw": False}
        if args.db:
            kwargs["db_path"] = args.db
        mem = NeuralMemory(**kwargs)
    print(captured.getvalue(), file=sys.stderr, end="")

    existing_hashes = _existing_content_hashes(mem.store)
    print(f"Already-ingested chunk count (by content_hash): {len(existing_hashes)}",
          file=sys.stderr)

    ingested = 0
    skipped_dup = 0
    failed = 0
    t0 = time.time()
    for i, c in enumerate(chunk_inventory):
        if c["content_hash"] in existing_hashes:
            skipped_dup += 1
            continue

        # Build memory content with section heading prefix for context
        content = f"[{c['heading']}]\n{c['body']}" if c["heading"] != "(intro)" else c["body"]
        label = f"{c['source_label']}:{c['path'].stem}:{c['heading'][:30]}"[:80]
        metadata = {
            "source_path": str(c["path"]),
            "source_label": c["source_label"],
            "section_heading": c["heading"],
            "content_hash": c["content_hash"],
            "file_mtime": c["mtime"],
            "ingested_by": "ingest_ae_corpus.py",
        }
        # Bridge messages carry extra structured metadata
        if c.get("_extra_metadata"):
            metadata.update(c["_extra_metadata"])

        try:
            mem.remember(
                content,
                label=label,
                detect_conflicts=False,  # we're ingesting durable canonical content
                kind=c["default_kind"],
                source=c["source_label"],
                origin_system=c["origin_system"],
                valid_from=c["mtime"],
                salience=args.salience,
                metadata=metadata,
            )
            ingested += 1
        except Exception as e:
            print(f"  fail {c['path'].name} [{c['heading']}]: {e}",
                  file=sys.stderr)
            failed += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(f"  progress: {i + 1}/{len(chunk_inventory)} "
                  f"(ingested={ingested}, dup={skipped_dup}, fail={failed}) "
                  f"rate={rate:.1f}/s",
                  file=sys.stderr)

    elapsed = time.time() - t0
    print(json.dumps({
        "total_chunks_seen": len(chunk_inventory),
        "ingested": ingested,
        "skipped_duplicate": skipped_dup,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 1),
        "rate_per_second": round(ingested / max(elapsed, 0.001), 2),
    }, indent=2))

    try:
        mem.close()
    except Exception:
        pass
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
