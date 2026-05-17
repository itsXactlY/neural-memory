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
    python -m benchmarks.external.longmemeval_s --recall-mode skynet \
        --rerank --backend auto --limit 50

Outputs JSON under ``benchmarks/external/results/longmemeval_s_<ts>.json``.

Canonical corpus source: ``benchmarks/snapshots/mm_bench_raw_20260513_0158_full.dump``
(regenerate via ``scripts/import_raw_corpora_to_pg.py``). Set
``MM_LME_FROM_PG=1`` to read from mm_bench_raw instead of the JSON file.
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

def _load_dataset_from_pg(variant: str) -> list[dict[str, Any]]:
    """Reconstruct the upstream JSON shape from mm_bench_raw.

    Why rebuild instead of selecting JSONB blobs? The importer split
    `haystack_sessions` into a relational `sessions` table for query-
    ability; we have to re-stitch by (question_id, session_idx, msg_idx)
    so existing harness code that walks haystack_sessions keeps working
    untouched.
    """
    import psycopg

    schema = f"longmemeval_{variant.lower()}"
    pw = os.environ.get("MM_POSTGRES_PASSWORD", "")
    host = os.environ.get("MM_POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("MM_POSTGRES_PORT", "5432")
    user = os.environ.get("MM_POSTGRES_USER", "mazemaker")
    dsn = os.environ.get("MM_POSTGRES_DSN") or (
        f"host={host} port={port} dbname=mm_bench_raw user={user} password={pw}"
    )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT question_id, question_type, question, question_date, "
                f"answer, answer_session_ids, haystack_dates, haystack_session_ids "
                f"FROM {schema}.questions"
            )
            q_rows = cur.fetchall()

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT question_id, session_idx, msg_idx, role, content "
                f"FROM {schema}.sessions "
                f"ORDER BY question_id, session_idx, msg_idx"
            )
            # bucket[(qid, s_idx)] = list[{role,content}] in msg_idx order
            bucket: dict[tuple[str, int], list[dict[str, Any]]] = {}
            for qid, s_idx, _m_idx, role, content in cur:
                bucket.setdefault((qid, s_idx), []).append(
                    {"role": role, "content": content}
                )

    records: list[dict[str, Any]] = []
    for (qid, qtype, qtext, qdate, ans, ans_ids, dates, sids) in q_rows:
        # answer_session_ids drives reconstruction order; haystack_session_ids
        # is the *full* haystack including distractors, so use that.
        full_sids = sids or []
        sessions = [bucket.get((qid, i), []) for i in range(len(full_sids))]
        records.append({
            "question_id": qid,
            "question_type": qtype,
            "question": qtext,
            "question_date": qdate,
            "answer": ans,
            "answer_session_ids": ans_ids or [],
            "haystack_dates": dates or [],
            "haystack_session_ids": full_sids,
            "haystack_sessions": sessions,
        })
    return records


def load_dataset(path: Path = DATASET_PATH, variant: str = "s") -> list[dict[str, Any]]:
    if os.environ.get("MM_LME_FROM_PG") == "1":
        return _load_dataset_from_pg(variant)
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
# Stratified sampling so every question_type appears proportionally
# ---------------------------------------------------------------------------

def stratified_sample(records: list[dict[str, Any]], limit: int,
                      seed: int = 42) -> list[dict[str, Any]]:
    import random
    rng = random.Random(seed)
    bucket: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        bucket.setdefault(r.get("question_type", "unknown"), []).append(r)
    # Proportionally allocate slots
    total = len(records)
    selected: list[dict[str, Any]] = []
    remaining = limit
    qtypes = sorted(bucket.keys(), key=lambda qt: -len(bucket[qt]))
    for i, qt in enumerate(qtypes):
        pool = bucket[qt]
        # Last type gets whatever's left; otherwise proportional
        if i == len(qtypes) - 1:
            n = remaining
        else:
            n = max(1, round(len(pool) / total * limit))
        n = min(n, len(pool), remaining)
        rng.shuffle(pool)
        selected.extend(pool[:n])
        remaining -= n
        if remaining <= 0:
            break
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Sliding-window chunker matching bake_chunked_sessions.py.

    Sessions <= chunk_chars stay whole. Above that, emit overlapping windows
    so every sentence is inside SOME chunk's ColBERT-encodable boundary.
    """
    if len(text) <= chunk_chars:
        return [text]
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        out.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return out


def ingest_session_level(nm: Mazemaker, record: dict[str, Any],
                         enable_chunks: bool = False) -> int:
    """Ingest each haystack session as one memory.

    When enable_chunks is on, each session is split into 2000-char windows
    with 200-char overlap; each chunk becomes its own memory labeled
    `session:<sid>::chunk::<n>`. The bench scorer matches via
    `label.split(':')` so the sid is still surfaced from every chunk.

    Returns number of memories stored.
    """
    sids = record.get("haystack_session_ids") or []
    sessions = record.get("haystack_sessions") or []
    count = 0
    for sid, msgs in zip(sids, sessions):
        # Concatenate all messages in this session into one text block
        text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in (msgs or [])
        )
        if not text.strip():
            continue
        if enable_chunks:
            for i, chunk in enumerate(_chunk_text(text)):
                nm.remember(chunk, label=f"session:{sid}::chunk::{i}",
                            auto_connect=False, detect_conflicts=False)
                count += 1
        else:
            nm.remember(text, label=f"session:{sid}",
                        auto_connect=False, detect_conflicts=False)
            count += 1
    return count


def ingest_turn_level(nm: Mazemaker, record: dict[str, Any]) -> int:
    """Ingest each chat turn as a separate memory.
    Returns number of memories stored."""
    sids = record.get("haystack_session_ids") or []
    sessions = record.get("haystack_sessions") or []
    count = 0
    for sid, msgs in zip(sids, sessions):
        for ti, m in enumerate(msgs or []):
            text = f"{m.get('role', 'user')}: {m.get('content', '')}"
            if not text.strip():
                continue
            nm.remember(text, label=f"turn:{sid}:{ti}",
                        auto_connect=False, detect_conflicts=False)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def rank_of_gold(results: list[dict[str, Any]],
                 gold_session_ids: set[str]) -> int | None:
    """Return 1-indexed rank of the first result matching a gold session,
    or None if none matched."""
    for i, r in enumerate(results, 1):
        label = r.get("label") or ""
        # Session-level: label is "session:<sid>" — check any label segment
        # matches a gold id.
        for segment in label.split(":"):
            if segment in gold_session_ids:
                return i
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def aggregate(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    gradeable = [q for q in per_question if q.get("gold_session_ids") and not q.get("is_abstention")]
    rrs: list[float] = []
    latencies: list[float] = []
    for q in gradeable:
        rank = q.get("rank_of_gold")
        if rank is not None and rank > 0:
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)
        if "latency_ms" in q:
            latencies.append(q["latency_ms"])
    n = len(gradeable)

    def _hits(k: int) -> int:
        return sum(1 for q in gradeable
                   if q.get("rank_of_gold") is not None and 0 < q["rank_of_gold"] <= k)

    ingest_values = [q["ingest_ms"] for q in per_question if "ingest_ms" in q]

    return {
        "n_total": len(per_question),
        "n_gradeable": n,
        "recall@1": round(_hits(1) / n, 4) if n else 0.0,
        "recall@5": round(_hits(5) / n, 4) if n else 0.0,
        "recall@10": round(_hits(10) / n, 4) if n else 0.0,
        "MRR": round(statistics.mean(rrs) if rrs else 0.0, 4),
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "mean_ingest_ms": round(statistics.mean(ingest_values), 1) if ingest_values else 0.0,
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
        return True


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _build_engine(args, db_path: str) -> Mazemaker:
    if getattr(args, "enable_colbert", False):
        os.environ["MM_COLBERT_ENABLED"] = "1"
    if getattr(args, "enable_dae", False):
        os.environ["MM_DAE_ENABLED"] = "1"
    channel_weights = None
    if getattr(args, "colbert_weight", None) is not None:
        channel_weights = {"colbert": float(args.colbert_weight)}
    if getattr(args, "dae_weight", None) is not None:
        channel_weights = channel_weights or {}
        channel_weights["dae"] = float(args.dae_weight)
    # Audit fix (2026-05-11): the published 97.87% R@5 LME-S run had
    # use_cpp=False, use_hnsw=False — Mazemaker's C++ kNN and HNSW
    # were off entirely. Both are now ON; expect a small recall+ and
    # a latency change. Rerank stays operator-controlled via --rerank
    # because it has a real perf cost (~8s cold-start per recall).
    return Mazemaker(
        db_path=db_path,
        embedding_backend=args.backend,
        use_cpp=True,
        retrieval_mode=args.recall_mode,
        use_hnsw="auto",
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

    # When PG backend is active, skip the SQLite temp file entirely.
    # No tempdir, no bench.db, no "no such table: memories" GPU cache
    # spam. PG handles everything.
    backend = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
    using_pg = (backend == "postgres")
    td_ctx = None
    if using_pg:
        db = ""
        nm = _build_engine(args, db)
    else:
        td_ctx = tempfile.TemporaryDirectory(prefix=f"lme-{qid[:8]}-")
        td = td_ctx.__enter__()
        db = str(Path(td) / "bench.db")
        nm = _build_engine(args, db)
    try:
        t0 = time.perf_counter()
        enable_chunks = getattr(args, "enable_chunks", False)
        if args.granularity == "turn":
            n_ingested = ingest_turn_level(nm, record)
        else:
            n_ingested = ingest_session_level(nm, record,
                                              enable_chunks=enable_chunks)
        ingest_ms = (time.perf_counter() - t0) * 1000.0

        # AFE second-pass (4th dream phase): extract atomic facts from each
        # long ingested session/chunk and add them as new memories with
        # `supports` back-edges. Runs in the per-question ephemeral engine
        # so the fact set is isolated to that question's haystack.
        if getattr(args, "enable_afe", False):
            try:
                from dream_engine import DreamEngine
                # AFE writes via the engine's _store_extracted_fact path;
                # backend dispatch picks SQLite or DreamPostgresStore from env.
                backend_choice = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
                if backend_choice == "postgres":
                    from dream_postgres_store import DreamPostgresStore
                    _be = DreamPostgresStore()
                else:
                    from dream_engine import SQLiteDreamBackend
                    _be = SQLiteDreamBackend(db)
                _de = DreamEngine(_be, neural_memory=nm)
                # Loop once — AFE processes up to MAZEMAKER_AFE_MAX_PER_CYCLE
                # per call. For per-question corpora (50-200 sessions) one
                # cycle is enough.
                _de._phase_afe()
            except Exception as _afe_err:
                print(f"[lme-s] WARN: AFE phase failed for {qid}: {_afe_err}",
                      file=sys.stderr, flush=True)

        # DAE second-pass: per-question ephemeral engines need the
        # memory_dae_embeddings table populated before recall can
        # consult the channel.  Failures here must not abort the
        # bench — log and continue with the primary embeddings.
        if getattr(args, "enable_dae", False):
            try:
                from dae import dae_bulk_compute
                dae_bulk_compute(
                    nm,
                    self_weight=float(args.dae_self_weight),
                    neighbour_k=int(args.dae_neighbour_k),
                )
            except Exception as _dae_err:
                print(f"[lme-s] WARN: dae_bulk_compute failed for "
                      f"{qid}: {_dae_err}", file=sys.stderr, flush=True)

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
        if getattr(args, "enable_dae", False):
            recall_kwargs["enable_dae"] = True
        if getattr(args, "dae_weight", None) is not None:
            recall_kwargs["dae_weight"] = float(args.dae_weight)
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
        if td_ctx is not None:
            try:
                td_ctx.__exit__(None, None, None)
            except Exception:
                pass


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
        selected: list[dict[str, Any]] = []
        remaining = args.limit
        qtypes = sorted(bucket.keys(), key=lambda qt: -len(bucket[qt]))
        for i, qt in enumerate(qtypes):
            pool = bucket[qt]
            n = remaining if i == len(qtypes) - 1 else max(1, round(len(pool) / len(records) * args.limit))
            n = min(n, len(pool), remaining)
            rng.shuffle(pool)
            selected.extend(pool[:n])
            remaining -= n
            if remaining <= 0:
                break
        rng.shuffle(selected)
        return selected
    elif args.limit:
        return records[: args.limit]
    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_result(args, per_question, metrics, qtype_metrics) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"longmemeval_s{tag}_{ts}.json"
    path = RESULTS_DIR / fname

    payload = {
        "timestamp": ts,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "system_config": {
            "recall_mode": args.recall_mode,
            "rerank": args.rerank,
            "embedding_backend": args.backend,
            "granularity": args.granularity,
            "k": args.k,
            "limit": args.limit,
            "use_hnsw": "auto",
            "use_cpp": True,
            "enable_colbert": bool(getattr(args, "enable_colbert", False)),
            "colbert_weight": getattr(args, "colbert_weight", None),
            "enable_dae": bool(getattr(args, "enable_dae", False)),
            "dae_weight": getattr(args, "dae_weight", None),
            "dae_self_weight": getattr(args, "dae_self_weight", None),
            "dae_neighbour_k": getattr(args, "dae_neighbour_k", None),
            "enable_afe": bool(getattr(args, "enable_afe", False)),
            "enable_chunks": bool(getattr(args, "enable_chunks", False)),
        },
        "dataset": {
            "name": "longmemeval_s",
            "path": str(DATASET_PATH),
            "sha256": dataset_hash(DATASET_PATH),
        },
        "metrics": metrics,
        "metrics_by_question_type": qtype_metrics,
        "per_question": per_question,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recall-mode", default="hybrid",
                    choices=["semantic", "hybrid", "advanced", "skynet", "lean", "trim"])
    p.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker")
    p.add_argument("--backend", default="auto", help="Embedding backend")
    p.add_argument("--granularity", default="session", choices=["session", "turn"])
    p.add_argument("-k", type=int, default=10, help="Top-k cutoff")
    p.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    p.add_argument("--stratified", action="store_true", help="Stratified sample when --limit is set")
    p.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling")
    p.add_argument("--tag", default="", help="Tag for output filename")
    p.add_argument("--enable-colbert", action="store_true")
    p.add_argument("--colbert-weight", type=float, default=None)
    p.add_argument("--enable-dae", action="store_true")
    p.add_argument("--dae-weight", type=float, default=None)
    p.add_argument("--dae-self-weight", type=float, default=0.4)
    p.add_argument("--dae-neighbour-k", type=int, default=20)
    p.add_argument("--enable-afe", action="store_true",
                   help="4th dream phase — extract atomic facts from each "
                        "ingested session/chunk before recall. Each fact becomes "
                        "a new memory linked back to source via `supports` edge.")
    p.add_argument("--enable-chunks", action="store_true",
                   help="Split each haystack session into 2000-char windows "
                        "with 200-char overlap before ingest (session-granularity "
                        "only; ignored under --granularity turn). Labels become "
                        "session:<sid>::chunk::<n>.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    records = load_dataset(DATASET_PATH, variant="s")
    selected = select_records(records, args)

    print(f"[lme-s] dataset={DATASET_PATH.name} total={len(records)} selected={len(selected)}", flush=True)
    if not args.quiet:
        print(f"[lme-s] config: recall_mode={args.recall_mode} "
              f"rerank={args.rerank} backend={args.backend} "
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
