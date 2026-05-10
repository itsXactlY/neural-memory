#!/usr/bin/env python3
"""Merge multiple demolition bench result files into one canonical file.

For each model, picks the run with the FEWEST errors (tiebreak: highest
accuracy, then most recent). Useful when an Ollama 500 storm during a
full grid run nuked some models — re-run them in isolation and merge.

Usage:
    python -u benchmarks/external/_merge_demolition.py \
        results/demolition_full-grid-synthetic20_*.json \
        results/demolition_rerun-2models_*.json \
        --tag canonical-synthetic20
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the markdown renderer from the harness
import demolition_bench as db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+",
                   help="Result JSON files (or globs). Later files take "
                        "priority on ties.")
    p.add_argument("--tag", default="canonical")
    args = p.parse_args()

    paths: list[Path] = []
    for pat in args.inputs:
        if any(c in pat for c in "*?["):
            paths.extend(Path(p_) for p_ in glob.glob(pat))
        else:
            paths.append(Path(pat))
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 1

    # Load all + pick best run per model
    runs: list[dict] = []
    for p_ in paths:
        runs.append(json.loads(p_.read_text()))

    best_models: dict[str, dict] = {}
    best_rows: dict[str, list] = {}
    chosen_run: dict[str, str] = {}
    for run in runs:
        ts = run.get("timestamp", "")
        for model, agg in run.get("models", {}).items():
            cand = (agg.get("errors", 999), -agg.get("accuracy", 0.0), ts)
            cur = best_models.get(model)
            cur_key = (
                cur.get("errors", 999),
                -cur.get("accuracy", 0.0),
                chosen_run.get(model, ""),
            ) if cur else (10**6, 0, "")
            if cand < cur_key:
                best_models[model] = agg
                best_rows[model] = run.get("per_question", {}).get(model, [])
                chosen_run[model] = ts

    # Use the most-recent run as the source of system_config + dataset
    base_run = max(runs, key=lambda r: r.get("timestamp", ""))
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": ts,
        "git_sha": base_run.get("git_sha", "unknown"),
        "git_dirty": base_run.get("git_dirty", False),
        "system_config": base_run.get("system_config", {}),
        "dataset": base_run.get("dataset", {}),
        "models": best_models,
        "per_question": best_rows,
        "_merged_from": [str(p_) for p_ in paths],
        "_chosen_run_per_model": chosen_run,
    }

    json_path = RESULTS_DIR / f"demolition_{args.tag}_{ts}.json"
    md_path = RESULTS_DIR / f"demolition_{args.tag}_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2))

    # Build a tiny shim args object the renderer can read
    class _A:
        pass
    a = _A()
    a.k_retrieval = payload["system_config"].get("k_retrieval", 10)

    md_path.write_text(db._render_markdown(payload, a))

    print(f"merged {len(runs)} runs into:")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")
    print("\nchosen run per model (timestamp):")
    for m, t in chosen_run.items():
        agg = best_models[m]
        print(f"  {m:<24} from {t}  "
              f"{agg['correct']}/{agg['total']}  errors={agg['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
