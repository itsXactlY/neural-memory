"""Mazemaker reference runner — drives the existing longmemeval_s.py
harness in a subprocess, then projects its output JSON into the
canonical `ResultRecord` shape.

Why subprocess instead of importing? The existing harness already
handles per-question DB tear-down, ColBERT env wiring, and
per-question-type aggregation correctly. Re-implementing that here
would risk silent drift in the headline number. Shelling out keeps
this runner thin and the harness authoritative.

Verified reference (master @ 443011cb on 2026-05-10):
  LongMemEval-S 500q, hybrid + ColBERT@1.5
  R@1=0.8574  R@5=0.9787  R@10=0.9894  MRR=0.9114
  p50=56.86ms  p95=60.79ms
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from runners.common import (
    DatasetMeta,
    Metrics,
    POD_VERSION,
    ResultRecord,
    base_argparser,
    host_info,
    load_dataset,
    now_iso,
    pod_root,
    write_result,
)

SYSTEM = "mazemaker"

# Path to the parent repo's external benchmark harness.
DEFAULT_MAZEMAKER_REPO = Path(__file__).resolve().parents[2]
HARNESS_REL = Path("benchmarks") / "external" / "longmemeval_s.py"


def _harness_path(repo: Path) -> Path:
    return repo / HARNESS_REL


def _engine_version(repo: Path) -> str:
    """Best-effort engine version: git SHA, falling back to 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode().strip()
        dirty = ""
        try:
            d = subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            if d.decode().strip():
                dirty = "-dirty"
        except Exception:
            pass
        return f"git:{sha}{dirty}"
    except Exception:
        return "unknown"


def _find_result_json(results_dir: Path, since_mtime: float) -> Path | None:
    """The harness writes longmemeval_s_<tag>_<ts>.json. We pick up the
    newest file produced after `since_mtime`."""
    cands = []
    for p in results_dir.glob("longmemeval_s_*.json"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > since_mtime:
            cands.append((mt, p))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def run(args) -> int:
    work = Path(args.work).expanduser().resolve()
    repo = Path(args.mazemaker_repo).expanduser().resolve()
    harness = _harness_path(repo)
    if not harness.exists():
        print(f"[mazemaker] harness not found: {harness}", file=sys.stderr)
        print(f"[mazemaker] pass --mazemaker-repo=<path> to override.", file=sys.stderr)
        return 2

    # Validate dataset and capture canonical hash up-front. The harness
    # also hashes it internally; we record both for the comparator.
    _, dataset_meta = load_dataset(work, pod_root(), args.dataset)

    results_dir = harness.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    start_mtime = time.time() - 1.0  # slight backdate so a same-second write is picked up

    cmd = [
        sys.executable,
        str(harness),
        "--recall-mode", args.recall_mode,
        "--backend", args.backend,
        "--granularity", args.granularity,
        "-k", str(args.k),
        "--tag", "bench-pod",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.enable_colbert:
        cmd.append("--enable-colbert")
    if args.colbert_weight is not None:
        cmd += ["--colbert-weight", str(args.colbert_weight)]
    if args.rerank:
        cmd.append("--rerank")
    if args.quiet:
        cmd.append("--quiet")

    if not args.quiet:
        print(f"[mazemaker] launching: {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(repo))
    wall = time.perf_counter() - t0

    if rc != 0:
        print(f"[mazemaker] harness exited with rc={rc}", file=sys.stderr)
        # Emit a partial record so the comparator can surface ERROR
        rec = ResultRecord(
            timestamp=now_iso(),
            bench_pod_version=POD_VERSION,
            system=SYSTEM,
            system_version=_engine_version(repo),
            system_config={"argv": cmd, "harness_rc": rc},
            dataset=dataset_meta,
            metrics=Metrics(errors=1),
            host=host_info(),
        )
        write_result(work, SYSTEM, rec)
        return rc

    result_json = _find_result_json(results_dir, start_mtime)
    if result_json is None:
        print("[mazemaker] could not locate harness output JSON", file=sys.stderr)
        return 3

    raw = json.loads(result_json.read_text())
    m = raw.get("metrics", {})
    cfg = raw.get("system_config", {})
    ds = raw.get("dataset", {})

    metrics = Metrics(
        r_at_1=_f(m.get("recall@1")),
        r_at_5=_f(m.get("recall@5")),
        r_at_10=_f(m.get("recall@10")),
        mrr=_f(m.get("MRR")),
        p50_recall_ms=_f(m.get("p50_ms")),
        p95_recall_ms=_f(m.get("p95_ms")),
        wall_seconds_ingest=_f(m.get("mean_ingest_ms"), scale=1.0 / 1000.0) if m.get("mean_ingest_ms") else None,
        wall_seconds_query=round(wall, 2),
        llm_tokens_extraction=None,   # Mazemaker uses embeddings, no LLM extraction step
        errors=sum(1 for q in raw.get("per_question", []) if q.get("error")),
        failed_questions=[i for i, q in enumerate(raw.get("per_question", []))
                          if q.get("error")],
    )

    # Cross-check dataset hash against the manifest-verified copy
    harness_hash = (ds.get("sha256") or "")[:16]
    if harness_hash and dataset_meta.hash and harness_hash != dataset_meta.hash:
        print(
            f"[mazemaker] WARNING: dataset hash from harness ({harness_hash}) "
            f"!= manifest ({dataset_meta.hash}); recording both.",
            file=sys.stderr,
        )

    rec = ResultRecord(
        timestamp=now_iso(),
        bench_pod_version=POD_VERSION,
        system=SYSTEM,
        system_version=_engine_version(repo),
        system_config={
            "recall_mode": cfg.get("recall_mode"),
            "rerank": cfg.get("rerank"),
            "embedding_backend": cfg.get("embedding_backend"),
            "granularity": cfg.get("granularity"),
            "k": cfg.get("k"),
            "enable_colbert": cfg.get("enable_colbert"),
            "colbert_weight": cfg.get("colbert_weight"),
            "harness_result": str(result_json),
        },
        dataset=dataset_meta,
        metrics=metrics,
        host=host_info(),
    )
    out_path = write_result(work, SYSTEM, rec)
    if not args.quiet:
        print(f"[mazemaker] wrote {out_path}", flush=True)
        print(f"[mazemaker] R@1={metrics.r_at_1} R@5={metrics.r_at_5} "
              f"R@10={metrics.r_at_10} MRR={metrics.mrr} p50={metrics.p50_recall_ms}ms",
              flush=True)
    return 0


def _f(v, scale: float = 1.0) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v) * scale, 6)
    except (TypeError, ValueError):
        return None


def build_parser():
    p = base_argparser(SYSTEM, "Mazemaker reference runner — drives benchmarks/external/longmemeval_s.py")
    p.add_argument("--mazemaker-repo", default=str(DEFAULT_MAZEMAKER_REPO),
                   help="Path to the mazemaker engine repo (default: parent of bench-pod/).")
    p.add_argument("--recall-mode", default="hybrid",
                   choices=["semantic", "hybrid", "advanced", "skynet", "lean", "trim"])
    p.add_argument("--backend", default="auto")
    p.add_argument("--granularity", default="session", choices=["session", "turn"])
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--enable-colbert", action="store_true", default=True,
                   help="Enable ColBERT late-interaction rerank channel (default: on — this is the reference config).")
    p.add_argument("--no-colbert", dest="enable_colbert", action="store_false")
    p.add_argument("--colbert-weight", type=float, default=1.5,
                   help="ColBERT channel weight (default: 1.5, the verified reference).")
    p.add_argument("--rerank", action="store_true", default=False)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
