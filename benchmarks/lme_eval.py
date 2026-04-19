#!/usr/bin/env python3
"""lme_eval.py — LongMemEval-style retrieval benchmark for neural-memory.

Measures Recall@k, MRR, and p50/p95 latency on a corpus of stored facts and
paraphrased queries.

Two modes:

  1. Real LongMemEval corpus (if a JSON/JSONL dataset is present on disk).
     Download from https://huggingface.co/datasets/xiaowu0162/long_mem_eval
     and pass --dataset /path/to/longmemeval_s.jsonl (or _m, _oracle, etc.).
     Each record is expected to carry:
        - memory_context: list of prior-turn facts (seeded into the store)
        - query: the test question
        - answer: ground-truth memory_id(s) or keyword set

  2. Synthetic smoke benchmark (default). Generates a small facts+queries
     corpus so the eval runs offline without downloads. Useful for:
        - sanity-checking patches
        - relative comparisons across engine={bfs,ppr}, rerank={on,off},
          use_hnsw={on,off}

Run:
    python3 benchmarks/lme_eval.py
    python3 benchmarks/lme_eval.py --engine ppr --rerank
    python3 benchmarks/lme_eval.py --dataset /path/to/longmemeval_s.jsonl

Reports Recall@{1,5,10}, MRR, p50/p95 retrieval latency, and emits a JSON
summary at --out (default stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make `python/` importable whether run from repo root or benchmarks/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from memory_client import NeuralMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic corpus — stands in for LongMemEval short split
# ---------------------------------------------------------------------------

# Each entry: (fact, paraphrased_query)
# Designed so recall@1 is meaningful only if semantic matching works —
# query wording doesn't lexically overlap fact wording much.
_SYNTHETIC = [
    ("The user's dog is named Lou", "what is the user's pet called"),
    ("The user's main project is Angels Electric, an electrical contracting company in Aurora IL",
     "where does the user run their business"),
    ("Casco Maizi is the Paperclip CEO who drives six worker agents",
     "who is in charge of paperclip"),
    ("The dashboard is served on port 4173 and the paperclip service on port 3101",
     "which network ports do the services listen on"),
    ("OTTO HARDENING.rtf is the canonical QBO integration doctrine",
     "what document defines the QuickBooks integration rules"),
    ("Lennar income arrives as ACH deposits, not invoices",
     "how does Lennar money come in"),
    ("The owner's name is Ernesto 'Tito' Valencia Godinez",
     "what is the business owner's full name"),
    ("Keller Farm is a Lennar Urban Townhomes + Andare site, not commercial",
     "what kind of project is Keller Farm"),
    ("The financial calendar is stored at Financial Calendar - Master.xlsm on OneDrive",
     "where is the master spreadsheet located"),
    ("Tito's wife owns a standalone Hermes VM with her own personality wizard",
     "who owns the hermes-standalone agent"),
    ("Vibha Choudhury is the only real remodel customer with five invoices totalling $11.2K",
     "which remodel client has actual invoice history"),
    ("The merge freeze begins 2026-03-05 for the mobile release cut",
     "when does the mobile team cut their release"),
    ("Production QBO was polluted with 22 fake invoices and 25 fake estimates from a probe run",
     "how much synthetic data contaminates QuickBooks"),
    ("Pulse-hermes ships with Brave, Serper, and Exa web backends plus NewsAPI",
     "which search providers does pulse support"),
    ("Neural-memory uses modern attention-based Hopfield pattern completion in its C++ layer",
     "what is the associative recall mechanism in neural-memory"),
]


def load_dataset(path: str | None) -> list[dict]:
    """Load a LongMemEval-style JSON/JSONL dataset, or the synthetic corpus."""
    if path:
        records = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        if p.suffix == ".jsonl":
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with p.open() as fh:
                records = json.load(fh)
        return records

    # Synthetic fallback
    return [
        {"fact": f, "query": q, "expected_keyword": f.split()[0].lower()}
        for f, q in _SYNTHETIC
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked_ids: list[int], gold_id: int, k: int) -> int:
    return int(gold_id in ranked_ids[:k])


def reciprocal_rank(ranked_ids: list[int], gold_id: int) -> float:
    for i, r in enumerate(ranked_ids, start=1):
        if r == gold_id:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_dataset(args.dataset)

    # Fresh DB per run — don't pollute the user's memory store
    db = args.db or str(Path(args.workdir) / "lme_eval.db")
    if os.path.exists(db):
        os.remove(db)

    mem = NeuralMemory(
        db_path=db,
        embedding_backend=args.embedding_backend,
        use_cpp=args.use_cpp,
        use_hnsw=args.use_hnsw,
        rerank=args.rerank,
    )

    # Seed the store and track gold (fact -> memory_id)
    gold: dict[int, int] = {}
    for i, rec in enumerate(records):
        fact = rec.get("fact") or rec.get("memory") or ""
        if not fact:
            continue
        mid = mem.remember(fact, label=f"lme_{i}")
        gold[i] = mid

    # Evaluate
    r1 = r5 = r10 = 0
    rr_sum = 0.0
    lats_ms: list[float] = []
    failures: list[dict] = []

    for i, rec in enumerate(records):
        query = rec.get("query") or rec.get("question") or ""
        if not query or i not in gold:
            continue

        t0 = time.perf_counter()
        results = mem.recall(query, k=max(10, args.k))
        lats_ms.append((time.perf_counter() - t0) * 1000.0)

        ranked_ids = [r["id"] for r in results]
        gid = gold[i]
        r1 += recall_at_k(ranked_ids, gid, 1)
        r5 += recall_at_k(ranked_ids, gid, 5)
        r10 += recall_at_k(ranked_ids, gid, 10)
        rr_sum += reciprocal_rank(ranked_ids, gid)

        if gid not in ranked_ids[: args.k] and args.show_failures:
            failures.append({
                "query": query,
                "gold_fact": rec.get("fact"),
                "top": [(r["label"], round(r["combined"], 3)) for r in results[:3]],
            })

    n = sum(1 for i, rec in enumerate(records) if rec.get("query") and i in gold)
    if n == 0:
        mem.close()
        raise RuntimeError("no records had both a fact and a query")

    lats_ms.sort()
    p50 = lats_ms[len(lats_ms) // 2]
    p95 = lats_ms[min(len(lats_ms) - 1, int(len(lats_ms) * 0.95))]

    report = {
        "dataset": args.dataset or "synthetic",
        "n_records": n,
        "embedding_backend": args.embedding_backend,
        "use_cpp": args.use_cpp,
        "use_hnsw": args.use_hnsw,
        "rerank": args.rerank,
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "mrr": round(rr_sum / n, 4),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
    }
    if args.show_failures:
        report["failures"] = failures

    mem.close()
    return report


def main():
    ap = argparse.ArgumentParser(description="LongMemEval-style retrieval benchmark")
    ap.add_argument("--dataset", default=None, help="Path to LongMemEval JSON/JSONL. Omit for synthetic.")
    ap.add_argument("--workdir", default="/tmp", help="Workdir for eval DB")
    ap.add_argument("--db", default=None, help="Explicit DB path (default: workdir/lme_eval.db)")
    ap.add_argument("--embedding-backend", default="auto")
    ap.add_argument("--use-cpp", action="store_true", default=False)
    ap.add_argument("--use-hnsw", action="store_true", default=True)
    ap.add_argument("--no-hnsw", dest="use_hnsw", action="store_false")
    ap.add_argument("--rerank", action="store_true", default=False)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-failures", action="store_true", default=False)
    ap.add_argument("--out", default=None, help="Write JSON report to this path (default: stdout)")
    args = ap.parse_args()

    report = run(args)
    blob = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(blob + "\n")
    else:
        print(blob)


if __name__ == "__main__":
    main()
