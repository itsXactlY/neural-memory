"""
Neural Memory Benchmark — Report Generator
===========================================
Parses benchmark results and produces human-readable + JSON reports.

Generates:
  - Console summary (ASCII)
  - JSON report (machine-readable)
  - Markdown report (for GitHub/GitLab)
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON."""
    if not results_path.exists():
        return {}
    return json.loads(results_path.read_text())


def color_score(score: float, inverse: bool = False) -> str:
    """Return ANSI color for a score. Green=good, red=bad."""
    if inverse:
        score = 1.0 - score
    if score >= 0.9:
        return "\033[92m"   # green
    elif score >= 0.7:
        return "\033[93m"   # yellow
    elif score >= 0.5:
        return "\033[33m"   # orange
    else:
        return "\033[91m"   # red


def reset_color() -> str:
    return "\033[0m"


def format_number(n: float, suffix: str = "") -> str:
    """Format a number with thousands separator."""
    if isinstance(n, float):
        if abs(n) >= 1000:
            return f"{n:,.1f}{suffix}"
        return f"{n:.2f}{suffix}"
    return str(n)


def section(title: str, width: int = 70) -> str:
    line = "─" * width
    return f"\n{line}\n  {title}\n{line}"


class ReportGenerator:
    def __init__(self, results: Dict[str, Any], output_dir: Optional[Path] = None):
        self.results = results
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"

    # ── Retrieval ───────────────────────────────────────────────────────────

    def render_retrieval(self, data: Dict) -> str:
        lines = []
        lines.append(section("RETRIEVAL BENCHMARK"))

        modes = data.get("modes", {})
        if not modes:
            lines.append("  No data available.")
            return "\n".join(lines)

        # Comparison table header
        lines.append(f"\n  {'Mode':<15} {'R@5':>6} {'MRR@5':>7} {'p50ms':>7} {'QPS':>8} {'Status'}")
        lines.append(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")

        for mode, mdata in modes.items():
            q = mdata.get("queries", {})
            l = mdata.get("latency", {})
            t = mdata.get("throughput", {})

            r5 = q.get("recall@5", {}).get("recall", 0.0)
            m5 = q.get("recall@5", {}).get("mrr", 0.0)
            lat = l.get("p50_ms", 0.0)
            qps = t.get("queries_per_second", 0.0)

            c_r = color_score(r5)
            c_m = color_score(m5)
            c_l = color_score(1 - min(lat / 500, 1))  # lower is better
            c_q = color_score(min(qps / 100, 1))

            lines.append(
                f"  {mode:<15} "
                f"{c_r}{r5:>6.3f}{reset_color()} "
                f"{c_m}{m5:>7.3f}{reset_color()} "
                f"{c_l}{lat:>6.1f}ms{reset_color()} "
                f"{c_q}{qps:>7.1f}{reset_color()}"
            )

        # Summary
        summary = data.get("summary", {})
        if summary.get("best_mode"):
            lines.append(f"\n  BEST MODE: {summary['best_mode']} "
                          f"(MRR={summary['modes'].get(summary['best_mode'], {}).get('mrr@5', '?')})")

        return "\n".join(lines)

    # ── Dream Engine ───────────────────────────────────────────────────────

    def render_dream(self, data: Dict) -> str:
        lines = []
        lines.append(section("DREAM ENGINE BENCHMARK"))

        phases = data.get("phases", {})
        if not phases:
            lines.append("  No phase data available.")
            return "\n".join(lines)

        lines.append(f"\n  {'Phase':<10} {'Time (s)':>10} {'Details'}")
        lines.append(f"  {'-'*10} {'-'*10} {'-'*40}")

        for phase, pdata in phases.items():
            elapsed = pdata.get("elapsed_s", 0)
            details = ", ".join(f"{k}={v}" for k, v in pdata.items() if k != "elapsed_s")
            lines.append(f"  {phase.upper():<10} {elapsed:>10.2f}s   {details[:40]}")

        deltas = data.get("deltas", {})
        lines.append(f"\n  Delta after dream consolidation:")
        lines.append(f"    Connections: {deltas.get('connections_delta', '?'):+d}")
        lines.append(f"    Isolated:    {deltas.get('isolated_delta', '?'):+d}")
        lines.append(f"    Recall@5:    {deltas.get('recall_delta', '?'):+.4f}")

        return "\n".join(lines)

    # ── GPU ────────────────────────────────────────────────────────────────

    def render_gpu(self, data: Dict) -> str:
        lines = []
        lines.append(section("GPU RECALL BENCHMARK"))

        gvc = data.get("gpu_vs_cpu", {})
        if not gvc:
            lines.append("  No GPU data available.")
            return "\n".join(lines)

        lines.append(f"\n  {'Backend':<10} {'QPS':>8} {'ms/query':>10} {'Store time':>12}")
        lines.append(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*12}")

        for backend, bdata in gvc.items():
            if "error" in bdata:
                lines.append(f"  {backend:<10} ERROR: {bdata['error'][:30]}")
            else:
                lines.append(
                    f"  {backend:<10} "
                    f"{bdata.get('qps', 0):>7.1f}  "
                    f"{bdata.get('ms_per_query', 0):>9.2f}ms  "
                    f"{bdata.get('store_time_s', 0):>11.2f}s"
                )

        return "\n".join(lines)

    # ── Scalability ────────────────────────────────────────────────────────

    def render_scalability(self, data: Dict) -> str:
        lines = []
        lines.append(section("SCALABILITY BENCHMARK"))

        tiers = data.get("tiers", {})
        if not tiers:
            lines.append("  No tier data available.")
            return "\n".join(lines)

        lines.append(f"\n  {'Tier':>8} {'Insert/s':>10} {'Recall/s':>10} {'ms/q':>7} {'DB (MB)':>8} {'WAL (MB)':>9}")
        lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*9}")

        for tier, tdata in sorted(tiers.items()):
            rate = tdata.get("insert_rate_per_s", 0)
            recall_rate = tdata.get("recall_rate_per_s", 0)
            ms_q = tdata.get("ms_per_query", 0)
            ss = tdata.get("sqlite_stats", {})
            db_mb = ss.get("db_size_mb", 0)
            wal_mb = ss.get("wal_size_mb", 0)
            lines.append(
                f"  {tier:>8,} "
                f"{rate:>10,.0f} "
                f"{recall_rate:>10,.0f} "
                f"{ms_q:>6.1f}ms "
                f"{db_mb:>7.1f}  "
                f"{wal_mb:>8.2f}"
            )

        summary = data.get("summary", {})
        deg = summary.get("degradation_ratio", 0)
        c = color_score(deg)
        lines.append(f"\n  Degradation ratio: {c}{deg:.3f}{reset_color()} "
                     f"(<0.5 = significant degradation at scale)")

        return "\n".join(lines)

    # ── Graph ──────────────────────────────────────────────────────────────

    def render_graph(self, data: Dict) -> str:
        lines = []
        lines.append(section("GRAPH TRAVERSAL BENCHMARK"))

        bfs = data.get("bfs", {})
        think = data.get("think", {})

        if bfs:
            lines.append(f"\n  BFS Traversal:")
            lines.append(f"  {'Depth':>6} {'Visited':>8} {'Time (ms)':>10}")
            lines.append(f"  {'-'*6} {'-'*8} {'-'*10}")
            for key, bdata in sorted(bfs.items()):
                depth = key.split("_")[-1]
                lines.append(
                    f"  {depth:>6} "
                    f"{bdata.get('total_visited', 0):>8,} "
                    f"{bdata.get('total_time_ms', 0):>9.1f}ms"
                )

        if think:
            lines.append(f"\n  Spreading Activation (think):")
            lines.append(f"  {'Depth':>6} {'Activated':>10} {'Time (ms)':>10}")
            lines.append(f"  {'-'*6} {'-'*10} {'-'*10}")
            for key, tdata in sorted(think.items()):
                depth = key.split("_")[-1]
                lines.append(
                    f"  {depth:>6} "
                    f"{tdata.get('total_activated', 0):>10,} "
                    f"{tdata.get('time_ms', 0):>9.1f}ms"
                )

        return "\n".join(lines)

    # ── Concurrent ─────────────────────────────────────────────────────────

    def render_concurrent(self, data: Dict) -> str:
        lines = []
        lines.append(section("CONCURRENT / WAL BENCHMARK"))

        ws = data.get("writer_scaling", {})
        rs = data.get("reader_scaling", {})
        mx = data.get("mixed", {})

        if ws:
            lines.append(f"\n  Writer Scaling:")
            lines.append(f"  {'Workers':>8} {'Ops/s':>10} {'ms/op':>8} {'Errors':>7}")
            lines.append(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*7}")
            for key, wdata in sorted(ws.items()):
                lines.append(
                    f"  {key:>8} "
                    f"{wdata.get('ops_per_second', 0):>10,.0f} "
                    f"{wdata.get('ms_per_op', 0):>7.3f}ms "
                    f"{len(wdata.get('errors', [])):>7}"
                )

        if mx:
            lines.append(f"\n  Mixed Workload:")
            lines.append(f"  {'Config':>8} {'Ops/s':>10} {'WAL (MB)':>9}")
            lines.append(f"  {'-'*8} {'-'*10} {'-'*9}")
            for key, mdata in sorted(mx.items()):
                lines.append(
                    f"  {key:>8} "
                    f"{mdata.get('ops_per_second', 0):>10,.0f} "
                    f"{mdata.get('wal_size_mb', 0):>8.2f}"
                )

        return "\n".join(lines)

    # ── Conflict ──────────────────────────────────────────────────────────

    def render_conflict(self, data: Dict) -> str:
        lines = []
        lines.append(section("CONFLICT DETECTION BENCHMARK"))

        sup = data.get("supersession", {})
        lines.append(f"\n  Conflict pairs stored:    {sup.get('conflict_pairs_stored', '?')}")
        lines.append(f"  Superseded detected:     {sup.get('superseded_detected', '?')}")

        rq = data.get("recall_quality", {})
        if rq:
            correct = sum(1 for v in rq.values() if v.get("correct"))
            lines.append(f"\n  Recall correctness: {correct}/{len(rq)} "
                         f"({correct/len(rq)*100:.0f}%)")

        sal = data.get("salience", {})
        if sal:
            sl = sal.get("sample_saliences", {})
            lines.append(f"\n  Salience (sample): min={sl.get('min', 0):.3f}, "
                         f"mean={sl.get('mean', 0):.3f}, max={sl.get('max', 0):.3f}")

        return "\n".join(lines)

    # ── Agentic ───────────────────────────────────────────────────────────

    def render_agentic(self, data: Dict) -> str:
        lines = []
        lines.append(section("AGENTIC WORKFLOW BENCHMARK"))

        summary = data.get("summary", {})
        lines.append(f"\n  Sessions run:       {summary.get('num_sessions', '?')}")
        lines.append(f"  Total actions:     {summary.get('total_actions', '?')}")
        lines.append(f"  Avg throughput:     {summary.get('avg_actions_per_second', '?'):.1f} actions/s")
        lines.append(f"  Min/Max:           {summary.get('min_aps', '?')}/{summary.get('max_aps', '?')} actions/s")

        agg = summary.get("action_aggregates", {})
        if agg:
            lines.append(f"\n  Per-action latency:")
            lines.append(f"  {'Action':>10} {'Avg (ms)':>10} {'p95 (ms)':>10}")
            lines.append(f"  {'-'*10} {'-'*10} {'-'*10}")
            for action, adata in agg.items():
                lines.append(
                    f"  {action:>10} "
                    f"{adata.get('avg_mean_ms', 0):>10.2f} "
                    f"{adata.get('avg_p95_ms', 0):>10.2f}"
                )

        return "\n".join(lines)

    # ── Full render ────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render complete console report."""
        lines = []

        # Header
        meta = self.results.get("meta", {})
        lines.append(f"""
╔══════════════════════════════════════════════════════════╗
║       NEURAL MEMORY BENCHMARK — RESULTS REPORT           ║
╠══════════════════════════════════════════════════════════╣
║  Started:  {meta.get('started_at', '?'):<42}║
║  Finished: {meta.get('finished_at', '?') if meta.get('finished_at') else 'running':<42}║
║  Runtime:  {meta.get('total_elapsed_s', '?'):>8.1f}s{' '*33}║
╚══════════════════════════════════════════════════════════╝""")

        # Suite renderers
        suites = self.results.get("suites", {})
        renderers = {
            "retrieval": self.render_retrieval,
            "dream": self.render_dream,
            "gpu": self.render_gpu,
            "scalability": self.render_scalability,
            "graph": self.render_graph,
            "concurrent": self.render_concurrent,
            "conflict": self.render_conflict,
            "agentic": self.render_agentic,
        }

        errors = self.results.get("errors", {})

        for name, sdata in suites.items():
            status = sdata.get("status", "?")
            elapsed = sdata.get("elapsed_s", 0)

            if status == "error":
                print(f"\n[ERROR] {name}: {errors.get(name, 'unknown error')}")
                continue

            renderer = renderers.get(name)
            if renderer:
                try:
                    result = sdata.get("result", {})
                    lines.append(renderer(result))
                    lines.append(f"  [Completed in {elapsed:.1f}s]")
                except Exception as e:
                    lines.append(f"\n  ERROR rendering {name}: {e}")

        # Errors summary
        if errors:
            lines.append(section("ERRORS"))
            for suite, err in errors.items():
                lines.append(f"  {suite}: {err}")

        return "\n".join(lines)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save all report formats."""
        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Console
        console_path = out_dir / "report_console.txt"
        console_path.write_text(self.render())
        print(f"  Console report: {console_path}")

        # JSON (already done by benchmark, but include metadata)
        json_path = out_dir / "report.json"
        meta = self.results.get("meta", {})
        meta["report_generated_at"] = datetime.now().isoformat()
        json_path.write_text(json.dumps(self.results, indent=2, default=str))
        print(f"  JSON report: {json_path}")

        return json_path


def generate_report(results_path: Path, output_dir: Optional[Path] = None) -> str:
    """Main entry point for report generation."""
    results = load_results(results_path)
    if not results:
        print(f"No results found at {results_path}")
        return ""

    rg = ReportGenerator(results, output_dir)
    output = rg.render()
    print(output)
    rg.save(output_dir)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("results", nargs="?", type=Path,
                        default=Path.home() / ".neural_memory_benchmark" / "results" / "full_benchmark_results.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    generate_report(args.results, args.output_dir)
