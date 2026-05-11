"""Comparator: reads per-system result JSON from $WORK/results/,
validates against the schema, asserts dataset-hash consistency, and
writes matrix.json / matrix.md / verdict.md to $WORK/.

Exit codes:
  0  success (matrix written; PENDING rows OK)
  2  dataset hash mismatch across results — hard abort
  3  no usable results at all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compare.schema import validate

SYSTEMS_ORDER = ["mazemaker", "hindsight", "letta", "mem0", "amem", "cognee"]


# ---------------------------------------------------------------------------
# Number formatting — never fabricate
# ---------------------------------------------------------------------------

def _fmt_metric(v: Any, kind: str = "ratio") -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if kind == "ratio":
        return f"{f:.4f}"
    if kind == "ms":
        return f"{f:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sys_name in SYSTEMS_ORDER:
        path = results_dir / f"{sys_name}.json"
        if not path.exists():
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception as e:
            print(f"[compare] {sys_name}: invalid JSON ({e}); rendering ERROR",
                  file=sys.stderr)
            out[sys_name] = {"_invalid": str(e)}
            continue
        errs = validate(rec)
        if errs:
            print(f"[compare] {sys_name}: schema errors: {errs}; rendering ERROR",
                  file=sys.stderr)
            rec.setdefault("_schema_errors", errs)
        out[sys_name] = rec
    return out


def _assert_dataset_consistency(records: dict[str, dict[str, Any]]) -> tuple[str, int, str]:
    """Return (dataset_name, dataset_size, dataset_hash); abort if results disagree."""
    seen: dict[str, dict[str, Any]] = {}
    for sys_name, rec in records.items():
        ds = (rec or {}).get("dataset") or {}
        if not ds.get("hash"):
            continue
        sig = (ds.get("name"), ds.get("size"), ds.get("hash"))
        seen[sys_name] = {"name": sig[0], "size": sig[1], "hash": sig[2]}
    if not seen:
        return ("unknown", 0, "—")

    hashes = {v["hash"] for v in seen.values()}
    if len(hashes) > 1:
        print("[compare][ABORT] dataset hash mismatch across results:", file=sys.stderr)
        for sys_name, sig in seen.items():
            print(f"  {sys_name}: {sig}", file=sys.stderr)
        print("Refusing to merge — bench numbers would be uncomparable.", file=sys.stderr)
        sys.exit(2)

    first = next(iter(seen.values()))
    return (first["name"], int(first["size"]), first["hash"])


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _build_row(system: str, rec: dict[str, Any] | None) -> dict[str, Any]:
    if rec is None:
        return {
            "system": system, "system_version": "—",
            "status": "PENDING", "r1": "PENDING", "r5": "PENDING",
            "r10": "PENDING", "mrr": "PENDING", "p50": "PENDING",
            "p95": "PENDING", "errors": "—",
        }
    if "_invalid" in rec or "_schema_errors" in rec:
        return {
            "system": system, "system_version": rec.get("system_version", "—"),
            "status": "ERROR", "r1": "ERROR", "r5": "ERROR",
            "r10": "ERROR", "mrr": "ERROR", "p50": "ERROR",
            "p95": "ERROR", "errors": "—",
        }
    m = rec.get("metrics", {})
    err_count = int(m.get("errors", 0) or 0)
    has_any = any(m.get(k) is not None for k in ("r_at_1", "r_at_5", "r_at_10", "mrr"))
    status = "OK" if has_any else "ERROR"
    return {
        "system": system,
        "system_version": rec.get("system_version", "—"),
        "status": status,
        "r1":  _fmt_metric(m.get("r_at_1")),
        "r5":  _fmt_metric(m.get("r_at_5")),
        "r10": _fmt_metric(m.get("r_at_10")),
        "mrr": _fmt_metric(m.get("mrr")),
        "p50": _fmt_metric(m.get("p50_recall_ms"), "ms"),
        "p95": _fmt_metric(m.get("p95_recall_ms"), "ms"),
        "errors": err_count,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_matrix_md(rows: list[dict[str, Any]], dataset_name: str,
                      dataset_size: int, dataset_hash: str) -> str:
    lines: list[str] = []
    lines.append(f"# Comparison Pod — Comparison Matrix")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}` (n={dataset_size}, hash=`{dataset_hash}`)")
    lines.append("")
    lines.append("| System | Version | Status | R@1 | R@5 | R@10 | MRR | p50 (ms) | p95 (ms) | Errors |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['system']}` | `{r['system_version']}` | {r['status']} | "
            f"{r['r1']} | {r['r5']} | {r['r10']} | {r['mrr']} | "
            f"{r['p50']} | {r['p95']} | {r['errors']} |"
        )
    lines.append("")
    lines.append("Legend: `OK` ran to completion; `PENDING` is a v0.1 stub awaiting v0.2 implementation; "
                 "`ERROR` is a runner that crashed or emitted a malformed record (see `logs/<system>.log`).")
    return "\n".join(lines) + "\n"


def _render_matrix_json(rows: list[dict[str, Any]], dataset_name: str,
                        dataset_size: int, dataset_hash: str,
                        records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": {"name": dataset_name, "size": dataset_size, "hash": dataset_hash},
        "rows": rows,
        "raw": records,
    }


def _render_verdict(rows: list[dict[str, Any]], dataset_name: str,
                    dataset_size: int, dataset_hash: str,
                    pod_root: Path, timestamp: str, pod_version: str) -> str:
    # Pick the OK row with the best R@1 (numeric, not "—")
    def _r1(row: dict[str, Any]) -> float:
        try:
            return float(row["r1"])
        except (ValueError, TypeError):
            return -1.0

    ok_rows = [r for r in rows if r["status"] == "OK" and _r1(r) >= 0]
    winner = None
    if ok_rows:
        ok_rows.sort(key=_r1, reverse=True)
        w = ok_rows[0]
        winner = {"system": w["system"], "r1": w["r1"], "mrr": w["mrr"]}

    pending = [r["system"] for r in rows if r["status"] == "PENDING"]
    errored = [r["system"] for r in rows if r["status"] == "ERROR"]

    ctx = {
        "timestamp": timestamp,
        "pod_version": pod_version,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "dataset_hash": dataset_hash,
        "rows": rows,
        "winner": winner,
        "pending_systems": pending,
        "error_systems": errored,
    }

    template_path = pod_root / "compare" / "verdict_template.md.j2"
    template_src = template_path.read_text()

    try:
        from jinja2 import Environment, BaseLoader, StrictUndefined
        env = Environment(loader=BaseLoader(), undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)
        tmpl = env.from_string(template_src)
        return tmpl.render(**ctx)
    except ImportError:
        # Minimal hand-rolled fallback — no control flow, just headline + table.
        out: list[str] = []
        out.append("# Comparison Pod — Verdict")
        out.append("")
        out.append(f"**Run:** {timestamp}")
        out.append(f"**Pod version:** `{pod_version}`")
        out.append(f"**Dataset:** `{dataset_name}` (n={dataset_size}, hash=`{dataset_hash}`)")
        out.append("")
        if winner:
            out.append(f"**Headline:** **{winner['system']}** leads on `R@1` at **{winner['r1']}** "
                       f"(MRR {winner['mrr']}).")
        else:
            out.append("**Headline:** No system produced a complete result this run.")
        out.append("")
        out.append("## Matrix")
        out.append("")
        out.append("| System | Version | Status | R@1 | R@5 | R@10 | MRR | p50 (ms) | p95 (ms) | Errors |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            out.append(
                f"| `{r['system']}` | `{r['system_version']}` | {r['status']} | "
                f"{r['r1']} | {r['r5']} | {r['r10']} | {r['mrr']} | "
                f"{r['p50']} | {r['p95']} | {r['errors']} |"
            )
        if pending:
            out.append("")
            out.append("Pending runners: " + ", ".join(f"`{s}`" for s in pending))
        if errored:
            out.append("Errored runners: " + ", ".join(f"`{s}`" for s in errored))
        out.append("")
        out.append("---")
        out.append("**Want this for your agents?**")
        out.append("Pro pod (Postgres + ColBERT@1.5 + Dream worker + Architect): https://mazemaker.online/onboard")
        out.append("Free during launch · Self-host · Single-tenant · No telemetry.")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import datetime as _dt

    p = argparse.ArgumentParser(description="Comparison Pod comparator")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pod-version", default=None)
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pod_root = Path(__file__).resolve().parents[1]

    pod_version = args.pod_version
    if not pod_version:
        vfile = pod_root / "VERSION"
        pod_version = vfile.read_text().strip() if vfile.exists() else "0.0.0"

    if not results_dir.exists():
        print(f"[compare] no results directory at {results_dir}", file=sys.stderr)
        return 3

    records = _load(results_dir)
    if not records:
        print("[compare] no result files found; rendering all rows as PENDING", file=sys.stderr)

    dataset_name, dataset_size, dataset_hash = _assert_dataset_consistency(records)

    rows = [_build_row(s, records.get(s)) for s in SYSTEMS_ORDER]

    matrix_md = _render_matrix_md(rows, dataset_name, dataset_size, dataset_hash)
    matrix_json = _render_matrix_json(rows, dataset_name, dataset_size, dataset_hash, records)

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    verdict = _render_verdict(rows, dataset_name, dataset_size, dataset_hash,
                              pod_root, timestamp, pod_version)

    (out_dir / "matrix.md").write_text(matrix_md)
    (out_dir / "matrix.json").write_text(json.dumps(matrix_json, indent=2, default=str))
    (out_dir / "verdict.md").write_text(verdict)

    print(f"[compare] wrote {out_dir / 'matrix.md'}")
    print(f"[compare] wrote {out_dir / 'matrix.json'}")
    print(f"[compare] wrote {out_dir / 'verdict.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
