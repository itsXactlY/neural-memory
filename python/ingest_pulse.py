#!/usr/bin/env python3
"""Ingest PULSE --emit for-memory JSON into Mazemaker.

Deterministic, idempotent, and cron-safe:
- exact dedup by stable label/dedup_key
- optional cosine dedup against existing pulse memories
- salience hints are written into SQLite
- cluster -> finding support edges are added as typed graph edges
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mazemaker import Memory  # noqa: E402


def _slug(value: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "pulse"


def _load_payload(path: str | None) -> dict[str, Any]:
    raw = Path(path).read_text() if path else sys.stdin.read()
    raw = raw.strip()
    if not raw:
        raise SystemExit("No input JSON provided")
    return json.loads(raw)


def _content_for_memory(topic: str, item: dict[str, Any]) -> str:
    entities = ", ".join(item.get("entities") or [])
    urls = "\n".join(f"- {u}" for u in (item.get("source_urls") or [])[:8])
    meta = item.get("metadata") or {}
    parts = [
        f"Dedup-Key: {item.get('dedup_key', '')}",
        f"Topic: {topic}",
        f"Kind: {item.get('kind', 'finding')}",
        item.get("content", "").strip(),
    ]
    if entities:
        parts.append(f"Entities: {entities}")
    if urls:
        parts.append(f"Source URLs:\n{urls}")
    if meta:
        compact_meta = {k: v for k, v in meta.items() if k in {"cluster_id", "source", "sources", "score", "final_score", "uncertainty"}}
        if compact_meta:
            parts.append("Metadata: " + json.dumps(compact_meta, sort_keys=True, default=str))
    return "\n".join(p for p in parts if p).strip()


def _set_salience(mem: Memory, mem_id: int, salience: float) -> None:
    try:
        store = mem._sqlite_memory.store  # intentional internal access for ingestion tuning
        store.conn.execute(
            "UPDATE memories SET salience = MAX(COALESCE(salience, 1.0), ?) WHERE id = ?",
            (float(salience), int(mem_id)),
        )
        store.conn.commit()
    except Exception:
        pass


def _find_exact(mem: Memory, label: str) -> list[dict[str, Any]]:
    try:
        return list(mem._sqlite_memory.store.find_by_label(label))
    except Exception:
        return []


def _near_duplicate_label(mem: Memory, content: str, threshold: float) -> str | None:
    if threshold <= 0:
        return None
    try:
        results = mem.recall(content[:700], k=3)
    except Exception:
        return None
    for r in results:
        label = r.get("label", "") or ""
        sim = float(r.get("similarity", r.get("combined", 0.0)) or 0.0)
        if label.startswith("pulse:") and sim >= threshold:
            return label
    return None


def ingest(payload: dict[str, Any], *, db_path: str | None = None, threshold: float = 0.72,
           dry_run: bool = False, auto_connect: bool = False) -> dict[str, Any]:
    if payload.get("schema") != "pulse.for-memory.v1":
        raise ValueError(f"Unsupported schema: {payload.get('schema')!r}")
    topic = payload.get("topic") or "pulse"
    memories = payload.get("memories") or []
    mem = Memory(
        db_path=db_path,
        embedding_backend=os.environ.get("NEURAL_MEMORY_EMBEDDING_BACKEND", "auto"),
        retrieval_mode=os.environ.get("NEURAL_MEMORY_RETRIEVAL_MODE", "hybrid"),
        use_hnsw=os.environ.get("NEURAL_MEMORY_USE_HNSW", "auto"),
        lazy_graph=os.environ.get("NEURAL_MEMORY_LAZY_GRAPH", "true").lower() in {"1", "true", "yes", "on"},
        think_engine=os.environ.get("NEURAL_MEMORY_THINK_ENGINE", "ppr"),
        rerank=os.environ.get("NEURAL_MEMORY_RERANK", "false").lower() in {"1", "true", "yes", "on"},
    )
    stats = {"seen": 0, "stored": 0, "updated": 0, "skipped": 0, "edges": 0, "ids": []}
    cluster_ids: dict[str, int] = {}
    finding_cluster: list[tuple[str, int, float]] = []
    try:
        for item in memories:
            stats["seen"] += 1
            label = item.get("label") or f"pulse:{_slug(topic)}:{stats['seen']}"
            content = _content_for_memory(topic, item)
            salience = float(item.get("salience", 1.0) or 1.0)
            exact = _find_exact(mem, label)
            target_label = label
            if exact:
                stats["updated"] += 1
            else:
                near = _near_duplicate_label(mem, content, threshold)
                if near:
                    target_label = near
                    stats["updated"] += 1
                else:
                    stats["stored"] += 1
            if dry_run:
                continue
            mem_id = mem.remember(content, label=target_label, auto_chunk=False,
                                  auto_connect=auto_connect, detect_conflicts=True)
            if isinstance(mem_id, list):
                mem_id = mem_id[0]
            _set_salience(mem, int(mem_id), salience)
            stats["ids"].append(int(mem_id))
            meta = item.get("metadata") or {}
            cluster_id = meta.get("cluster_id")
            if item.get("kind") == "cluster" and cluster_id:
                cluster_ids[str(cluster_id)] = int(mem_id)
            elif cluster_id:
                finding_cluster.append((str(cluster_id), int(mem_id), salience))

        if not dry_run:
            store = mem._sqlite_memory.store
            for cluster_id, finding_id, salience in finding_cluster:
                source_id = cluster_ids.get(cluster_id)
                if not source_id or source_id == finding_id:
                    continue
                try:
                    store.add_connection(source_id, finding_id, min(0.95, max(0.4, salience / 2.0)), edge_type="pulse_supports")
                    stats["edges"] += 1
                except Exception:
                    pass
        return stats
    finally:
        try:
            mem.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PULSE for-memory JSON into Mazemaker")
    parser.add_argument("input", nargs="?", help="JSON file. Defaults to stdin.")
    parser.add_argument("--db", dest="db_path", help="Mazemaker SQLite DB path")
    parser.add_argument("--threshold", type=float, default=0.72, help="Pulse-only cosine dedup threshold (0 disables)")
    parser.add_argument("--auto-connect", action="store_true", help="Enable full auto-connect scan for new memories")
    parser.add_argument("--dry-run", action="store_true", help="Parse and dedup without writing")
    args = parser.parse_args()
    payload = _load_payload(args.input)
    stats = ingest(payload, db_path=args.db_path, threshold=args.threshold,
                   dry_run=args.dry_run, auto_connect=args.auto_connect)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
