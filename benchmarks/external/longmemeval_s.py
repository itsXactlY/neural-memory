#!/usr/bin/env python3
"""LongMemEval-S external benchmark harness for Mazemaker.

Runs the official LongMemEval-S dataset (Wu et al. ICLR 2025) against a fresh
Mazemaker engine instance per question, then aggregates session-level recall.

Granularity
-----------
Each `haystack_session` is ingested as a single Mazemaker memory with label
``session:<haystack_session_id>``. The dataset's gold ``answer_session_ids``
list is then matched against the labels in the recall result list, so a "hit"
means Mazemaker surfaced one of the evidence sessions in its top-k.

Two granularities are supported via ``--granularity``:
  - ``session`` (default): one memory per session, labeled ``session:<sid>``
  - ``turn``: one memory per turn, labeled ``turn:<sid>:<turn_idx>``; a hit
    requires the surfaced turn to belong to a gold session.

Usage
-----
    python -m benchmarks.external.longmemeval_s --recall-mode hybrid --k 10
    python -m benchmarks.external.longmemeval_s --recall-mode skynet \\
        --rerank --backend auto --limit 50

Outputs JSON under ``benchmarks/external/results/longmemeval_s_<ts>.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from memory_client import Mazemaker  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "longmemeval_s"
DATASET_PATH = DATA_DIR / "longmemeval_s.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval-S dataset not found at {path}. See README.md "
            f"in benchmarks/external/ for download instructions."
        )
    with path.open("rb") as fh:
        raw = fh.read()
    return json.loads(raw.decode("utf-8"))


def dataset_hash(path: Path = DATASET_PATH) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def question_types(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("question_type", "unknown")] = counts.get(r.get("question_type", "unknown"), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def _format_session(sess: list[dict[str, Any]], date: str | None) -> str:
    """Render a session's turns into a single chunk of text.

    Format: ``[date] role: content`` separated by newlines.
    """
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")
    for turn in sess:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ingest_session_level(nm: Mazemaker, record: dict[str, Any]) -> int:
    """Ingest each session as a single memory; returns count ingested."""
    n = 0
    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)
    for sid, sess, date in zip(sids, sessions, dates):
        text = _format_session(sess, date)
        if not text.strip():
            continue
        nm.remember(
            text,
            label=f"session:{sid}",
            auto_connect=False,
            detect_conflicts=False,
        )
        n += 1
    return n


def ingest_turn_level(nm: Mazemaker, record: dict[str, Any]) -> int:
    n = 0
    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)
    for sid, sess, date in zip(sids, sessions, dates):
        for j, turn in enumerate(sess):
            content = turn.get("content", "")
            if not content.strip():
                continue
            prefix = f"[{date}] " if date else ""
            text = f"{prefix}{turn.get('role','?')}: {content}"
            nm.remember(
                text,
                label=f"turn:{sid}:{j}",
                auto_connect=False,
                detect_conflicts=False,
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _result_session_id(result: dict[str, Any]) -> Optional[str]:
    label = result.get("label") or ""
    if label.startswith("session:"):
        return label.split(":", 1)[1]
    if label.startswith("turn:"):
        # turn:<sid>:<idx>
        parts = label.split(":")
        if len(parts) >= 3:
            return parts[1]
    return None


def rank_of_gold(results: list[dict[str, Any]], gold_session_ids: set[str]) -> Optional[int]:
    """Return the 1-indexed rank of the first result whose session matches gold.

    For session-level granularity each session appears once, so the rank is
    direct. For turn-level we rank by FIRST surfaced turn from a gold session.
    Returns None if no gold session appears in the result list.
    """
    seen: set[str] = set()
    for i, r in enumerate(results, start=1):
        sid = _result_session_id(r)
        if sid is None:
            continue
        if sid in gold_session_ids and sid not in seen:
            return i
        seen.add(sid)
    return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[idx]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _build_engine(args, db_path: str) -> Mazemaker:
    # ColBERT opt-in. The flag flips MM_COLBERT_ENABLED in the env so
    # remember() will write the per-memory token blob during ingest;
    # without that, recall would have nothing to score against.
    if getattr(args, "enable_colbert", False):
        os.environ["MM_COLBERT_ENABLED"] = "1"
    channel_weights = None
    if getattr(args, "colbert_weight", None) is not None:
        channel_weights = {"colbert": float(args.colbert_weight)}
    return Mazemaker(
        db_path=db_path,
        embedding_backend=args.backend,
        use_cpp=False,
        retrieval_mode=args.recall_mode,
        use_hnsw=False,
        lazy_graph=True,
        think_engine="bfs",
        rerank=args.rerank,
        channel_weights=channel_weights,
    )


def run_question(args, record: dict[str, Any]) -> dict[str, Any]:
    qid = record["question_id"]
    question = record["question"]
    gold_ids = set(record.get("answer_session_ids") or [])
    is_abstention = qid.endswith("_abs")

    with tempfile.TemporaryDirectory(prefix=f"lme-{qid[:8]}-") as td:
        db = str(Path(td) / "bench.db")
        nm = _build_engine(args, db)
        try:
            t0 = time.perf_counter()
            if args.granularity == "turn":
                n_ingested = ingest_turn_level(nm, record)
            else:
                n_ingested = ingest_session_level(nm, record)
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            recall_kwargs = dict(
                k=args.k,
                hybrid=(args.recall_mode in {"hybrid", "advanced", "skynet", "lean", "trim"}),
                rerank=args.rerank,
            )
            if getattr(args, "enable_colbert", False):
                recall_kwargs["enable_colbert"] = True
            if getattr(args, "colbert_weight", None) is not None:
                recall_kwargs["colbert_weight"] = float(args.colbert_weight)
            results = nm.recall(question, **recall_kwargs)
            recall_ms = (time.perf_counter() - t1) * 1000.0

            rank = rank_of_gold(results, gold_ids) if gold_ids else None
            return {
                "qid": qid,
                "question_type": record.get("question_type"),
                "is_abstention": is_abstention,
                "n_ingested": n_ingested,
                "n_results": len(results),
                "gold_session_ids": sorted(gold_ids),
                "rank_of_gold": rank,
                "ingest_ms": round(ingest_ms, 2),
                "latency_ms": round(recall_ms, 2),
                "top_labels": [r.get("label", "") for r in results[: args.k]],
            }
        finally:
            try:
                nm.close()
            except Exception:
                pass


def aggregate(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    # Restrict metrics to questions that HAVE gold sessions (skip abstention,
    # which has no positive evidence to retrieve).
    gradeable = [q for q in per_question if q.get("gold_session_ids") and not q.get("is_abstention")]
    n = max(1, len(gradeable))

    def _hits(thresh: int) -> int:
        return sum(1 for q in gradeable if q.get("rank_of_gold") is not None and q["rank_of_gold"] <= thresh)

    rrs: list[float] = []
    for q in gradeable:
        r = q.get("rank_of_gold")
        rrs.append(1.0 / r if r else 0.0)

    latencies = [q["latency_ms"] for q in per_question if "latency_ms" in q]

    return {
        "n_total": len(per_question),
        "n_gradeable": len(gradeable),
        "recall@1": round(_hits(1) / n, 4),
        "recall@5": round(_hits(5) / n, 4),
        "recall@10": round(_hits(10) / n, 4),
        "MRR": round(statistics.mean(rrs) if rrs else 0.0, 4),
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "mean_ingest_ms": round(statistics.mean([q["ingest_ms"] for q in per_question if "ingest_ms" in q]) or 0.0, 1) if per_question else 0.0,
    }


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def write_result(args, per_question: list[dict[str, Any]], metrics: dict[str, Any], qtype_metrics: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"longmemeval_s_{ts}.json"
    if args.tag:
        name = f"longmemeval_s_{args.tag}_{ts}.json"
    path = RESULTS_DIR / name

    payload = {
        "timestamp": ts,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "system_config": {
            "recall_mode": args.recall_mode,
            "rerank": bool(args.rerank),
            "embedding_backend": args.backend,
            "granularity": args.granularity,
            "k": args.k,
            "limit": args.limit,
            "seed": args.seed,
            "stratified": bool(args.stratified),
            "use_hnsw": False,
            "use_cpp": False,
            "enable_colbert": bool(getattr(args, "enable_colbert", False)),
            "colbert_weight": getattr(args, "colbert_weight", None),
        },
        "dataset": {
            "name": "longmemeval_s",
            "path": str(DATASET_PATH),
            "sha256": dataset_hash() if DATASET_PATH.exists() else None,
        },
        "metrics": metrics,
        "metrics_by_question_type": qtype_metrics,
        "per_question": per_question,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def aggregate_by_qtype(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for q in per_question:
        bucket.setdefault(q.get("question_type", "unknown"), []).append(q)
    return {qt: aggregate(rows) for qt, rows in bucket.items()}


def select_records(records: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.stratified and args.limit and args.limit < len(records):
        # Sample proportionally per question_type so small classes still appear.
        import random
        rng = random.Random(args.seed)
        bucket: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            bucket.setdefault(r.get("question_type", "unknown"), []).append(r)
        total = len(records)
        out: list[dict[str, Any]] = []
        for qt, rows in bucket.items():
            n_take = max(1, round(len(rows) / total * args.limit))
            rng.shuffle(rows)
            out.extend(rows[:n_take])
        rng.shuffle(out)
        return out[: args.limit]
    if args.limit and args.limit < len(records):
        return records[: args.limit]
    return records


def main() -> int:
    p = argparse.ArgumentParser(description="LongMemEval-S external harness")
    p.add_argument("--recall-mode", default="hybrid",
                   choices=["semantic", "hybrid", "advanced", "skynet", "lean", "trim"],
                   help="retrieval_mode passed to Mazemaker(...)")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranker on the head of recall")
    p.add_argument("--backend", default="auto",
                   help="Embedding backend: auto (BGE-M3 if available), hash (fast smoke), fastembed, ...")
    p.add_argument("--granularity", default="session", choices=["session", "turn"])
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on number of questions (0 = all)")
    p.add_argument("--stratified", action="store_true",
                   help="When --limit < N, sample proportionally per question_type")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="Optional tag prefix on the result filename")
    p.add_argument("--dataset", default=str(DATASET_PATH))
    p.add_argument("--quiet", action="store_true")
    # ColBERT-style late-interaction rerank. Default OFF so existing
    # baseline numbers don't shift unintentionally; flip on via
    # --enable-colbert. The flag also flips MM_COLBERT_ENABLED=1 so the
    # per-question ingest actually populates the token-cache the rerank
    # reads from.
    p.add_argument("--enable-colbert", action="store_true",
                   help="Enable ColBERT late-interaction rerank channel")
    p.add_argument("--colbert-weight", type=float, default=None,
                   help="Override colbert channel weight (default: preset-driven; "
                        "skynet=1.2, advanced/hybrid=0.5, lean/trim/semantic=0)")
    args = p.parse_args()

    records = load_dataset(Path(args.dataset))
    selected = select_records(records, args)
    if not args.quiet:
        print(f"[lme-s] dataset={Path(args.dataset).name} total={len(records)} selected={len(selected)}", flush=True)
        print(f"[lme-s] config: recall_mode={args.recall_mode} rerank={args.rerank} backend={args.backend} "
              f"granularity={args.granularity} k={args.k}", flush=True)
        print(f"[lme-s] type counts (selected): {question_types(selected)}", flush=True)

    per_question: list[dict[str, Any]] = []
    t_run = time.perf_counter()
    for i, rec in enumerate(selected):
        try:
            row = run_question(args, rec)
        except Exception as e:
            row = {
                "qid": rec.get("question_id"),
                "question_type": rec.get("question_type"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "is_abstention": (rec.get("question_id", "").endswith("_abs")),
                "gold_session_ids": sorted(set(rec.get("answer_session_ids") or [])),
            }
        per_question.append(row)
        if not args.quiet and (i + 1) % max(1, len(selected) // 20) == 0:
            done = i + 1
            elapsed = time.perf_counter() - t_run
            mid = aggregate(per_question)
            print(f"[lme-s] {done}/{len(selected)}  R@1={mid['recall@1']:.3f} R@5={mid['recall@5']:.3f} "
                  f"R@10={mid['recall@10']:.3f} MRR={mid['MRR']:.3f}  "
                  f"p50={mid['p50_ms']:.1f}ms p95={mid['p95_ms']:.1f}ms  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    metrics = aggregate(per_question)
    qtype_metrics = aggregate_by_qtype(per_question)
    out_path = write_result(args, per_question, metrics, qtype_metrics)

    print("\n=== LongMemEval-S Results ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nresult file: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
