#!/usr/bin/env python3
"""lme_real.py — LongMemEval benchmark (real JSON format).

Unlike the synthetic lme_eval.py, this handles the actual LongMemEval dataset
shape: records carry `haystack_sessions` (multi-session conversation context)
and the task is to recall memories originating from the gold `answer_session_ids`
given a new `question`.

Each record is processed as:
  1. Flatten haystack_sessions → individual turns → remember() each with
     label `lme:{question_id}:{session_id}:{turn_idx}`
  2. mem.recall(question, k=K)
  3. Check: do the top-K returned memories' labels reference a session_id
     that's in answer_session_ids? If yes: hit.

Metrics: Recall@{1,5,10}, MRR, p50/p95 latency, per-record seed time.

Usage:
    python3 benchmarks/lme_real.py --dataset /tmp/lme/longmemeval_s --max 20
    python3 benchmarks/lme_real.py --dataset /tmp/lme/longmemeval_s --max 20 --rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from memory_client import NeuralMemory  # noqa: E402


def _flatten_haystack(record) -> list[tuple[str, str, int, str]]:
    """Return list of (question_id, session_id, turn_idx, turn_text)."""
    qid = record["question_id"]
    turns = []
    haystack = record.get("haystack_sessions", [])
    session_ids = record.get("haystack_session_ids", [])
    if len(haystack) != len(session_ids):
        # Some records might pair differently; align by index
        session_ids = session_ids[: len(haystack)] + [f"s{i}" for i in range(len(session_ids), len(haystack))]

    for sess_idx, session in enumerate(haystack):
        sess_id = session_ids[sess_idx] if sess_idx < len(session_ids) else f"s{sess_idx}"
        # Session is a list of turns, each turn is {role, content}
        for turn_idx, turn in enumerate(session):
            if isinstance(turn, dict):
                content = turn.get("content", "")
            else:
                content = str(turn)
            if content.strip():
                turns.append((qid, sess_id, turn_idx, content.strip()))
    return turns


def _label_for(qid, sess_id, turn_idx) -> str:
    return f"lme:{qid}:{sess_id}:{turn_idx}"


def _session_from_label(label: str) -> str:
    # lme:{qid}:{sess_id}:{turn}
    parts = label.split(":")
    if len(parts) >= 3:
        return parts[2]
    return ""


def run(args):
    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {dataset} ({dataset.stat().st_size / 1e6:.1f} MB)")
    with dataset.open() as fh:
        records = json.load(fh)
    print(f"  {len(records)} total records. Using first {args.max}.")
    records = records[: args.max]

    # Use isolated per-run DB so we don't pollute the live neural-memory store
    db = args.db or str(Path(args.workdir) / f"lme-real-{int(time.time())}.db")
    if os.path.exists(db):
        os.remove(db)
    print(f"  DB: {db}")

    mem = NeuralMemory(
        db_path=db,
        embedding_backend=args.embedding_backend,
        use_cpp=args.use_cpp,
        use_hnsw=args.use_hnsw,
        rerank=args.rerank,
    )
    print(f"  backend: {mem.stats()}")

    # Per-record: seed + query + score
    all_results = []
    latencies_ms = []
    seed_times_s = []

    for i, rec in enumerate(records):
        qid = rec["question_id"]
        question = rec.get("question", "")
        answer_sessions = set(rec.get("answer_session_ids", []))

        turns = _flatten_haystack(rec)
        if not turns or not question:
            continue

        # Seed haystack into memory
        t_seed = time.perf_counter()
        if args.no_auto_connect:
            # H14: bulk-seed bypass — skip mem.remember()'s auto-connect O(n)
            # scan + optionally batch the embedder for ~10x throughput.
            # Edges are absent for these memories — fine for retrieval-only
            # benchmarks; think()/walk semantics are unavailable on them.
            batch_size = args.batch_embed
            if batch_size > 0:
                for j in range(0, len(turns), batch_size):
                    chunk = turns[j:j + batch_size]
                    contents = [c[:2000] for _, _, _, c in chunk]
                    vecs = mem.embedder.embed_batch(contents)
                    for (qid_r, sess_id, turn_idx, _), text, vec in zip(chunk, contents, vecs):
                        mem.store.store(_label_for(qid_r, sess_id, turn_idx), text, vec)
            else:
                for qid_r, sess_id, turn_idx, content in turns:
                    text = content[:2000]
                    vec = mem.embedder.embed(text)
                    mem.store.store(_label_for(qid_r, sess_id, turn_idx), text, vec)
        else:
            for qid_r, sess_id, turn_idx, content in turns:
                mem.remember(content[:2000], label=_label_for(qid_r, sess_id, turn_idx))
        seed_times_s.append(time.perf_counter() - t_seed)

        # Query
        t_recall = time.perf_counter()
        results = mem.recall(question, k=max(10, args.k))
        latencies_ms.append((time.perf_counter() - t_recall) * 1000.0)

        # Score: rank of first memory whose session_id is in answer_sessions
        rank = None
        for i_r, r in enumerate(results, start=1):
            sess = _session_from_label(r.get("label", ""))
            if sess in answer_sessions:
                rank = i_r
                break

        all_results.append({
            "qid": qid,
            "question": question[:80],
            "answer_sessions": sorted(answer_sessions),
            "rank": rank,
            "turns_seeded": len(turns),
        })

        if (i + 1) % 5 == 0 or i == len(records) - 1:
            hits_at_1 = sum(1 for r in all_results if r["rank"] == 1)
            hits_at_5 = sum(1 for r in all_results if r["rank"] and r["rank"] <= 5)
            hits_at_10 = sum(1 for r in all_results if r["rank"] and r["rank"] <= 10)
            print(f"  [{i+1}/{len(records)}] R@1={hits_at_1}/{i+1}  R@5={hits_at_5}/{i+1}  R@10={hits_at_10}/{i+1}")

    # Aggregate
    n = len(all_results)
    r1 = sum(1 for r in all_results if r["rank"] == 1) / n if n else 0
    r5 = sum(1 for r in all_results if r["rank"] and r["rank"] <= 5) / n if n else 0
    r10 = sum(1 for r in all_results if r["rank"] and r["rank"] <= 10) / n if n else 0
    mrr = sum(1.0 / r["rank"] for r in all_results if r["rank"]) / n if n else 0

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0
    p95 = latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95))] if latencies_ms else 0

    total_seed_s = sum(seed_times_s)
    total_turns = sum(r["turns_seeded"] for r in all_results)

    report = {
        "dataset": str(dataset),
        "n_records": n,
        "n_turns_seeded": total_turns,
        "seed_total_s": round(total_seed_s, 2),
        "seed_per_turn_ms": round((total_seed_s / total_turns) * 1000, 2) if total_turns else 0,
        "embedding_backend": mem.stats().get("embedding_backend"),
        "use_cpp": args.use_cpp,
        "use_hnsw": args.use_hnsw,
        "rerank": args.rerank,
        "recall@1": round(r1, 4),
        "recall@5": round(r5, 4),
        "recall@10": round(r10, 4),
        "mrr": round(mrr, 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
    }

    mem.close()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/tmp/lme/longmemeval_s")
    ap.add_argument("--max", type=int, default=20, help="Cap records (default 20)")
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--db", default=None)
    ap.add_argument("--embedding-backend", default="auto")
    ap.add_argument("--use-cpp", action="store_true", default=False)
    ap.add_argument("--use-hnsw", action="store_true", default=True)
    ap.add_argument("--no-hnsw", dest="use_hnsw", action="store_false")
    ap.add_argument("--rerank", action="store_true", default=False)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    # H14: bench-only flags for tractable ST + rerank full runs
    ap.add_argument(
        "--no-auto-connect", action="store_true",
        help="Skip mem.remember() auto-connect during bulk seeding. "
             "Memories stored, edges absent. Retrieval-only benchmarks.",
    )
    ap.add_argument(
        "--batch-embed", type=int, default=0,
        help="Batch size for embedder.embed_batch() during seeding. "
             "0 = per-turn (current behavior). Requires --no-auto-connect "
             "since auto-connect is per-row anyway.",
    )
    args = ap.parse_args()

    report = run(args)
    blob = json.dumps(report, indent=2)
    print("\n" + "=" * 60)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob + "\n")


if __name__ == "__main__":
    main()
