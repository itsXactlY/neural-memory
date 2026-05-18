"""
Lean-Skynet Comparison Suite
============================
Codex's v5 verdict accepted the benchmark with two named caveats that
we address here:

  - "Latency is real. Skynet's p50 = 340ms is 200× the raw-cosine
    baseline."
  - "Weak channels remain. BM25 / temporal contribute nothing
    measurable on this dataset, salience is null-or-slightly-harmful.
    Production code should consider trimming or reweighting them."

The new `retrieval_mode="lean"` in Mazemaker is the engineering
response: it zeroes the channels that channel_ablation proved are
dead weight (bm25, temporal, salience), keeping only the channels
that contributed meaningfully (semantic, entity, ppr).

This suite measures whether `lean` actually delivers — same data,
same queries, same embedder, just toggling the retrieval_mode. We
expect:

  * lean.recall ≈ skynet.recall (within ~1 pt; the dropped channels
    contribute zero on this corpus per channel_ablation).
  * lean.p50 < skynet.p50 (likely 30-60% reduction since BM25/entity
    each fire one FTS query per call, and salience computation
    iterates the whole candidate pool).

If lean recall drops materially below skynet, the "dead weight"
finding was an artefact and `lean` should not be promoted as a
production default. If it holds, lean is the recommended skynet
variant for latency-sensitive callers.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from mazemaker import Memory


def _measure(mem: Memory, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    hits = 0
    rrs: List[float] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = mem.recall(q["query"], k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        anchor = q.get("anchor", "")
        rank = 0
        for i, r in enumerate(results, 1):
            if anchor and anchor in (r.get("content") or ""):
                rank = i
                break
        hits += 1 if rank > 0 else 0
        rrs.append(1.0 / rank if rank > 0 else 0.0)
    n = len(queries) or 1
    latencies_ms.sort()
    return {
        "recall_at_k": round(hits / n, 4),
        "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
        "p50_ms": round(latencies_ms[len(latencies_ms) // 2], 3),
        "p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95)], 3),
        "p99_ms": round(latencies_ms[int(len(latencies_ms) * 0.99)], 3),
        "n": n,
    }


class LeanSkynetBenchmark:
    """Same data, same queries — three modes (semantic / skynet / lean)."""

    def __init__(
        self,
        memories: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        k: int = 5,
    ):
        self.memories = memories
        self.queries = queries
        self.k = k
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build(self, mode: str) -> Memory:
        # Each mode gets a fresh tempfile DB so leftover state from a
        # prior build (HNSW indexes warmed by an earlier run, FTS
        # caches, etc.) cannot bias the latency measurement.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        mem = Memory(
            db_path=db_path,
            embedding_backend="auto",
            retrieval_mode=mode,
        )
        for m in self.memories:
            mem.remember(m["text"], label=m["label"], auto_connect=True)
        return mem

    def run(self) -> Dict[str, Any]:
        print("\n=== Lean-Skynet Comparison Benchmark ===")
        results: Dict[str, Any] = {"modes": {}}

        # Four modes: semantic (no fusion), skynet (full fusion),
        # lean (synthetic-tuned: drops bm25/temporal/salience),
        # trim (universally-safe: drops only salience).
        # codex v6 (2026-04-28) caught that lean over-generalises from
        # synthetic data — on real prose bm25/temporal contribute, but
        # salience is universally dead/harmful. trim is the safer default.
        for mode in ["semantic", "skynet", "lean", "trim"]:
            print(f"  [{mode}] building + querying...")
            mem = self._build(mode)
            stats = _measure(mem, self.queries, self.k)
            # Record the actual channel_weights the mode used so the
            # comparison is auditable (lean must show 0 for bm25/
            # temporal/salience).
            # F78 fix (audit 2026-05-13): walk attributes defensively so
            # the suite reports `unavailable` (not None) when the layout
            # shifts, instead of silently zeroing a load-bearing field.
            inner = getattr(mem, "_sqlite_memory", None) or mem
            cw = getattr(inner, "_channel_weights", None)
            if isinstance(cw, dict):
                stats["channel_weights"] = dict(cw)
            else:
                stats["channel_weights"] = "unavailable (engine internal layout changed)"
            results["modes"][mode] = stats
            print(f"    R@{self.k}={stats['recall_at_k']}  MRR={stats['mrr']}  "
                  f"p50={stats['p50_ms']}ms  p95={stats['p95_ms']}ms")

        # Cross-mode lifts and latency comparisons.
        sky = results["modes"]["skynet"]
        lean = results["modes"]["lean"]
        trim = results["modes"]["trim"]
        sem = results["modes"]["semantic"]

        def _vs(arm: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "recall_delta": round(arm["recall_at_k"] - baseline["recall_at_k"], 4),
                "mrr_delta": round(arm["mrr"] - baseline["mrr"], 4),
                "p50_speedup": (
                    round(baseline["p50_ms"] / arm["p50_ms"], 2)
                    if arm["p50_ms"] > 0 else None
                ),
                "p50_savings_ms": round(baseline["p50_ms"] - arm["p50_ms"], 3),
            }

        results["analysis"] = {
            "lean_vs_skynet": _vs(lean, sky),
            "trim_vs_skynet": _vs(trim, sky),
            "skynet_vs_semantic_p50_overhead_ms": round(sky["p50_ms"] - sem["p50_ms"], 3),
            "interpretation": (
                "lean is synthetic-tuned (drops bm25/temporal/salience); on "
                "real prose codex v6 verdict caught those channels do "
                "contribute. trim drops ONLY salience (universally dead per "
                "ablation) so it's the safer production default. Compare "
                "trim_vs_skynet on real corpora to decide; lean_vs_skynet on "
                "synthetic to see the latency ceiling."
            ),
        }
        print(f"\n  lean_vs_skynet:  Δrecall={results['analysis']['lean_vs_skynet']['recall_delta']:+.4f}  "
              f"speedup={results['analysis']['lean_vs_skynet']['p50_speedup']}× "
              f"({results['analysis']['lean_vs_skynet']['p50_savings_ms']}ms saved)")
        print(f"  trim_vs_skynet:  Δrecall={results['analysis']['trim_vs_skynet']['recall_delta']:+.4f}  "
              f"speedup={results['analysis']['trim_vs_skynet']['p50_speedup']}× "
              f"({results['analysis']['trim_vs_skynet']['p50_savings_ms']}ms saved)")

        out = self.output_dir / "lean_skynet_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
