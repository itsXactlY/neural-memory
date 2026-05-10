#!/usr/bin/env python3
"""Compare two LongMemEval-S result files and print a markdown table.

Usage:
    python -m benchmarks.external.compare_runs <baseline.json> <candidate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = ["recall@1", "recall@5", "recall@10", "MRR", "p50_ms", "p95_ms"]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _delta(b: float, c: float, key: str) -> str:
    d = c - b
    arrow = ""
    if key.endswith("_ms"):
        # lower is better
        arrow = "↓" if d < 0 else ("↑" if d > 0 else "·")
        return f"{d:+.2f} {arrow}"
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
    return f"{d:+.4f} {arrow}"


def render_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    bm = baseline.get("metrics", {})
    cm = candidate.get("metrics", {})
    bcfg = baseline.get("system_config", {})
    ccfg = candidate.get("system_config", {})

    lines: list[str] = []
    lines.append("# LongMemEval-S — run comparison\n")
    lines.append(f"- **Baseline**: `{baseline.get('git_sha','?')[:12]}` "
                 f"(mode={bcfg.get('recall_mode')}, rerank={bcfg.get('rerank')}, "
                 f"backend={bcfg.get('embedding_backend')}, granularity={bcfg.get('granularity')}, "
                 f"k={bcfg.get('k')}, n={bm.get('n_gradeable','?')})")
    lines.append(f"- **Candidate**: `{candidate.get('git_sha','?')[:12]}` "
                 f"(mode={ccfg.get('recall_mode')}, rerank={ccfg.get('rerank')}, "
                 f"backend={ccfg.get('embedding_backend')}, granularity={ccfg.get('granularity')}, "
                 f"k={ccfg.get('k')}, n={cm.get('n_gradeable','?')})\n")

    lines.append("| Metric | Baseline | Candidate | Δ |")
    lines.append("|---|---:|---:|---:|")
    for key in METRIC_KEYS:
        bv = bm.get(key, 0.0)
        cv = cm.get(key, 0.0)
        try:
            delta = _delta(float(bv), float(cv), key)
        except (TypeError, ValueError):
            delta = "—"
        lines.append(f"| {key} | {_fmt(bv)} | {_fmt(cv)} | {delta} |")

    # Per-question-type table
    bq = baseline.get("metrics_by_question_type", {})
    cq = candidate.get("metrics_by_question_type", {})
    types = sorted(set(bq) | set(cq))
    if types:
        lines.append("\n## By question_type — recall@5\n")
        lines.append("| question_type | Baseline R@5 | Candidate R@5 | Δ |")
        lines.append("|---|---:|---:|---:|")
        for qt in types:
            b = bq.get(qt, {}).get("recall@5", 0.0)
            c = cq.get(qt, {}).get("recall@5", 0.0)
            try:
                delta = _delta(float(b), float(c), "recall@5")
            except (TypeError, ValueError):
                delta = "—"
            lines.append(f"| {qt} | {_fmt(b)} | {_fmt(c)} | {delta} |")

    # Per-question rank movement
    pb = {q.get("qid"): q for q in baseline.get("per_question", [])}
    pc = {q.get("qid"): q for q in candidate.get("per_question", [])}
    common = sorted(set(pb) & set(pc))
    promoted: list[tuple[str, Any, Any]] = []
    demoted: list[tuple[str, Any, Any]] = []
    for qid in common:
        rb = pb[qid].get("rank_of_gold")
        rc = pc[qid].get("rank_of_gold")
        if rb == rc:
            continue
        # treat None as +inf
        rb_v = rb if rb is not None else 999
        rc_v = rc if rc is not None else 999
        if rc_v < rb_v:
            promoted.append((qid, rb, rc))
        elif rc_v > rb_v:
            demoted.append((qid, rb, rc))
    if promoted or demoted:
        lines.append(f"\n## Rank movement (top changes)\n")
        lines.append(f"- Promoted (gold moved up): **{len(promoted)}**")
        lines.append(f"- Demoted (gold moved down): **{len(demoted)}**\n")
        if promoted:
            lines.append("### Top 10 promotions")
            promoted.sort(key=lambda r: ((r[1] or 999) - (r[2] or 999)), reverse=True)
            lines.append("| qid | baseline rank | candidate rank |")
            lines.append("|---|---:|---:|")
            for qid, rb, rc in promoted[:10]:
                lines.append(f"| `{qid}` | {rb} | {rc} |")
        if demoted:
            lines.append("\n### Top 10 demotions")
            demoted.sort(key=lambda r: ((r[2] or 999) - (r[1] or 999)), reverse=True)
            lines.append("| qid | baseline rank | candidate rank |")
            lines.append("|---|---:|---:|")
            for qid, rb, rc in demoted[:10]:
                lines.append(f"| `{qid}` | {rb} | {rc} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two LongMemEval-S result JSONs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--out", default="", help="Optional file to write the markdown to")
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    md = render_table(baseline, candidate)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
