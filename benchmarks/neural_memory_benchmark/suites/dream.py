"""
Dream Engine Benchmark Suite (v2 — actually exercises DreamEngine)
==================================================================
The previous version was broken in ways that mattered:

  * It poked SQLiteDreamBackend.strengthen_connection(mid, mid) (self-loops),
    bypassing DreamEngine._phase_nrem entirely. NREM in the real system fires
    spreading activation, collects activated edges, and bulk-strengthens
    them; the prior bench measured none of that.
  * `_run_insight` had a `pass` inside its loop and returned a hard-coded
    `communities_detected: 0`. Louvain was never invoked.
  * `start_session()` returns an int but the bench did `stats["session_id"]`
    — the suite would crash at line 1 if it ran.
  * `measure_connection_stats` read keys (`memory_count`, `connection_count`,
    `density`, `isolated_count`, `community_count`) that don't exist in the
    actual `nm.graph()` shape (`{nodes, edges, top_edges}`); every baseline
    silently reported zero.

This rewrite calls the real entrypoints — `DreamEngine.dream_now()` for the
end-to-end pipeline, and the individual `_phase_*` methods when phases are
selected — and measures the actual deltas the dream pipeline is supposed to
produce: connection growth, isolated-memory reduction, community formation,
and a recall-quality lift on a held-out paraphrase query set.

Metrics:
  * pre vs post graph stats (nodes, edges, top edges by weight)
  * NREM:    activated_edges_strengthened, weak_pruned, weakened
  * REM:     bridges_inserted (only counts NEW edges, not pre-existing)
  * Insight: communities_found, derived_facts_materialised
  * Recall@5 / MRR delta on paraphrase queries
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory
from dream_engine import DreamEngine, SQLiteDreamBackend


# ─────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────

def _graph_snapshot(db_path: str) -> Dict[str, int]:
    """Snapshot the connection graph directly from SQLite, not via
    NeuralMemory.graph() — that method has a different schema and was the
    source of the prior `0` baselines."""
    out: Dict[str, int] = {
        "memories": 0,
        "connections": 0,
        "isolated": 0,
        "bridges": 0,
        "derived_facts": 0,
        "max_weight_edges": 0,
    }
    try:
        conn = sqlite3.connect(db_path)
        out["memories"] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        out["connections"] = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        out["isolated"] = conn.execute(
            "SELECT COUNT(*) FROM memories m WHERE NOT EXISTS "
            "(SELECT 1 FROM connections c WHERE c.source_id = m.id OR c.target_id = m.id)"
        ).fetchone()[0]
        out["bridges"] = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE COALESCE(edge_type,'similar') = 'bridge'"
        ).fetchone()[0]
        out["derived_facts"] = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE label LIKE 'derived:%'"
        ).fetchone()[0]
        # >0.7 weight edges are the "consolidated" ones — should grow during NREM
        # if anything is being strengthened, drop during REM only if pruning fires.
        out["max_weight_edges"] = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE weight >= 0.7"
        ).fetchone()[0]
        conn.close()
    except Exception:
        pass
    return out


def _recall_quality(nm: NeuralMemory, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """Recall@k + MRR on the paraphrase query set.

    queries must carry `ground_truth_ids` (list[str]) and the dataset id
    is the `paraphrase:<topic>` label-ish string. We translate via label
    lookup since the synthetic ids aren't preserved in DB rowids.
    """
    if not queries:
        return {"recall@k": 0.0, "mrr": 0.0, "n": 0}

    hits = []
    rrs = []
    for q in queries:
        results = nm.recall(q["query"], k=k)
        # Match on the unique anchor token — only the right statement contains it.
        anchor = q.get("anchor", "")
        rank = 0
        for i, r in enumerate(results, 1):
            if anchor and anchor in (r.get("content") or ""):
                rank = i
                break
        hits.append(1.0 if rank > 0 else 0.0)
        rrs.append(1.0 / rank if rank > 0 else 0.0)

    return {
        "recall@k": round(statistics.mean(hits), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "n": len(queries),
    }


# ─────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────

class DreamBenchmark:
    """Exercises DreamEngine end-to-end and measures the deltas it
    actually produces, not just the calls it makes."""

    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        test_queries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        phases: Optional[List[str]] = None,
    ):
        self.db_path = db_path
        self.memories = memories
        self.test_queries = test_queries
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.phases = phases or ["nrem", "rem", "insight"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nm: Optional[NeuralMemory] = None
        self.engine: Optional[DreamEngine] = None

    # -- setup ----------------------------------------------------------------

    def setup(self) -> Dict[str, Any]:
        print(f"  [setup] storing {len(self.memories)} memories...")
        self.nm = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        # auto_connect=True so the graph has structure for NREM to consolidate
        # — without edges the dream engine has nothing to strengthen.
        for i, m in enumerate(self.memories, 1):
            self.nm.remember(m["text"], label=m["label"], auto_connect=True)
            if i % 500 == 0:
                print(f"    {i}/{len(self.memories)}")

        # DreamEngine wraps the same SQLite store the NeuralMemory just wrote to.
        self.engine = DreamEngine.sqlite(
            self.db_path,
            neural_memory=self.nm,
            idle_threshold=10**9,   # disable auto-fire; we drive cycles manually
            memory_threshold=10**9,
        )
        return _graph_snapshot(self.db_path)

    # -- run ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        print("\n=== Dream Engine Benchmark (v2 — real engine) ===")
        results: Dict[str, Any] = {}

        baseline_graph = self.setup()
        baseline_recall = _recall_quality(self.nm, self.test_queries)
        print(f"  [baseline] graph: {baseline_graph}")
        print(f"  [baseline] recall: {baseline_recall}")

        results["baseline"] = {"graph": baseline_graph, "recall": baseline_recall}
        results["phases"] = {}

        # Run each phase individually so we can attribute deltas. The phases
        # must run in NREM → REM → Insight order; running them out of order
        # changes the semantics (REM bridges feed Insight communities).
        order = ["nrem", "rem", "insight"]
        selected = [p for p in order if p in self.phases]

        prev_graph = baseline_graph
        for phase in selected:
            print(f"\n  === phase: {phase.upper()} ===")
            t0 = time.perf_counter()
            phase_stats = self._run_phase(phase)
            elapsed = time.perf_counter() - t0
            post_graph = _graph_snapshot(self.db_path)
            results["phases"][phase] = {
                "elapsed_s": round(elapsed, 3),
                "phase_stats": phase_stats,
                "graph_after": post_graph,
                "deltas_vs_prev_phase": {
                    "connections": post_graph["connections"] - prev_graph["connections"],
                    "isolated": post_graph["isolated"] - prev_graph["isolated"],
                    "bridges": post_graph["bridges"] - prev_graph["bridges"],
                    "max_weight_edges": (
                        post_graph["max_weight_edges"] - prev_graph["max_weight_edges"]
                    ),
                    "derived_facts": post_graph["derived_facts"] - prev_graph["derived_facts"],
                },
            }
            print(f"    elapsed={elapsed:.2f}s  stats={phase_stats}")
            print(f"    deltas={results['phases'][phase]['deltas_vs_prev_phase']}")
            prev_graph = post_graph

        post_recall = _recall_quality(self.nm, self.test_queries)
        post_graph = _graph_snapshot(self.db_path)
        results["post_dream"] = {"graph": post_graph, "recall": post_recall}
        results["overall_deltas"] = {
            "connections": post_graph["connections"] - baseline_graph["connections"],
            "isolated": post_graph["isolated"] - baseline_graph["isolated"],
            "bridges": post_graph["bridges"] - baseline_graph["bridges"],
            "derived_facts": post_graph["derived_facts"] - baseline_graph["derived_facts"],
            "recall_at_k_delta": round(
                post_recall["recall@k"] - baseline_recall["recall@k"], 4
            ),
            "mrr_delta": round(post_recall["mrr"] - baseline_recall["mrr"], 4),
        }
        print(f"\n  [overall] {results['overall_deltas']}")

        self.save(results)
        return results

    def _run_phase(self, phase: str) -> Dict[str, Any]:
        """Drive the real DreamEngine phase, not a hand-rolled stub."""
        if phase == "nrem":
            return self.engine._phase_nrem()
        if phase == "rem":
            return self.engine._phase_rem()
        if phase == "insight":
            return self.engine._phase_insights()
        return {"error": f"unknown phase {phase}"}

    def save(self, results: Dict[str, Any]) -> Path:
        out = self.output_dir / "dream_results.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  [saved] {out}")
        return out
