"""
Mazemaker Benchmark — Main Orchestrator
===========================================
Coordinates all benchmark suites and collects results.

Usage:
    python benchmark.py                    # run all suites
    python benchmark.py --suite retrieval   # run specific suite
    python benchmark.py --dry-run          # generate data only

Note:
    This module is loaded by `runner.py`, which inserts BENCH_ROOT (the
    `neural_memory_benchmark/` directory) at the front of sys.path. Local
    imports like ``from config import ...`` therefore resolve to the
    benchmark-local files (``config.py``, ``dataset.py``, ``suites/...``).
    Do NOT add the project's ``python/`` directory to sys.path here — it
    contains a separate ``config.py`` that would shadow the bench config.
    Each suite handles its own ``python/`` import for ``memory_client``.
"""
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BenchmarkConfig, HOT_PATH_DB
from dataset import MasterDataset, QueryGenerator


def _assert_not_hotpath(p) -> None:
    """Refuse to run if any path resolves to the production hot-path DB.

    Override only with NEURAL_BENCH_ALLOW_HOTPATH=1 (an explicit acknowledgment
    that you really want to write into ~/.neural_memory/memory.db).
    """
    if os.environ.get("NEURAL_BENCH_ALLOW_HOTPATH") == "1":
        return
    try:
        resolved = Path(p).expanduser().resolve()
    except Exception:
        return
    if resolved == HOT_PATH_DB:
        raise RuntimeError(
            f"Refusing to run benchmark against the production hot-path DB: {resolved}. "
            f"Set NEURAL_BENCH_ALLOW_HOTPATH=1 to override (you almost never want this)."
        )


ALL_SUITES = [
    "retrieval", "dream", "gpu", "scalability",
    "graph", "concurrent", "conflict", "agentic", "qa",
    # v2 suites — added for the "truly unique" pass:
    "diversity", "lstm_knn", "continuity", "conflict_quality", "baseline",
    # v3 suites — codex audit 2026-04-28 follow-ups:
    "graph_reasoning", "channel_ablation", "hnsw_exactness",
    "dream_derived_fact", "continuity_controls",
    # v5 suites — codex 2026-04-28 caveat fixes:
    "lean_skynet",
]


def banner(text: str, width: int = 70) -> str:
    line = "=" * width
    pad = (width - len(text) - 2) // 2
    return f"\n{line}\n{' ' * pad} {text}\n{line}\n"


# ── Suite dispatch ───────────────────────────────────────────────────────────
# Each entry constructs a <Name>Benchmark instance and calls .run().
# Suites that have a module-level `run_<name>_benchmark` function are routed
# through it for backward compatibility.

def _new_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", prefix="nmb-", delete=False) as f:
        path = f.name
    _assert_not_hotpath(path)
    return path


def _run_retrieval(cfg, memories, queries):
    from suites.retrieval import RetrievalBenchmark
    bm = RetrievalBenchmark(
        db_path=_new_db(),
        memories=memories,
        queries=queries,
        modes=cfg.retrieval.modes,
        top_ks=cfg.retrieval.top_ks,
        latency_runs=cfg.retrieval.latency_runs,
        output_dir=cfg.paths.results_dir,
    )
    results = bm.run()
    bm.save(results)
    return results


def _run_dream(cfg, memories, queries):
    from suites.dream import DreamBenchmark
    bm = DreamBenchmark(
        db_path=_new_db(),
        memories=memories,
        test_queries=queries,
        output_dir=cfg.paths.results_dir,
        phases=cfg.dream.phases,
    )
    return bm.run()


def _run_gpu(cfg, memories, queries):
    from suites.gpu import GPUBenchmark
    bm = GPUBenchmark(
        memories=memories,
        output_dir=cfg.paths.results_dir,
        batch_sizes=cfg.gpu.batch_sizes,
    )
    return bm.run()


def _run_scalability(cfg, memories, queries):
    from suites.scalability import ScalabilityBenchmark
    bm = ScalabilityBenchmark(
        memories=memories,
        tiers=cfg.scalability.tiers,
        output_dir=cfg.paths.results_dir,
    )
    return bm.run()


def _run_graph(cfg, memories, queries):
    from suites.graph import GraphBenchmark
    bm = GraphBenchmark(
        db_path=_new_db(),
        memories=memories,
        output_dir=cfg.paths.results_dir,
        depths=cfg.graph.depths,
    )
    return bm.run()


def _run_concurrent(cfg, memories, queries):
    from suites.concurrent import ConcurrentBenchmark
    bm = ConcurrentBenchmark(
        memories=memories,
        output_dir=cfg.paths.results_dir,
    )
    return bm.run()


def _run_conflict(cfg, memories, queries):
    from suites.conflict import ConflictBenchmark
    bm = ConflictBenchmark(
        db_path=_new_db(),
        memories=memories,
        output_dir=cfg.paths.results_dir,
        conflict_groups=cfg.conflict.conflict_groups,
    )
    return bm.run()


def _run_agentic(cfg, memories, queries):
    from suites.agentic import AgenticBenchmark
    bm = AgenticBenchmark(
        db_path=_new_db(),
        memories=memories,
        output_dir=cfg.paths.results_dir,
        num_sessions=cfg.agentic.sessions,
        turns_per_session=cfg.agentic.turns_per_session,
    )
    return bm.run()


def _run_qa(cfg, memories, queries):
    from suites.qa import QABenchmark
    # The QA suite is intentionally driven by its own pre-canned (question,
    # context, needle) facts — distractors come from the synthetic dataset.
    bm = QABenchmark(
        db_path=_new_db(),
        memories=memories,
        output_dir=cfg.paths.results_dir,
        modes=cfg.retrieval.modes if cfg.retrieval.modes else ["semantic", "hybrid"],
        top_k=5,
    )
    return bm.run()


def _run_diversity(cfg, memories, queries):
    from suites.diversity import DiversityBenchmark
    bm = DiversityBenchmark(
        db_path=_new_db(),
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
    )
    return bm.run()


def _run_lstm_knn(cfg, memories, queries):
    from suites.lstm_knn import LSTMKnnBenchmark
    bm = LSTMKnnBenchmark(
        db_path=_new_db(),
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
    )
    return bm.run()


def _run_continuity(cfg, memories, queries):
    from suites.continuity import ContinuityBenchmark
    bm = ContinuityBenchmark(
        db_path=_new_db(),
        output_dir=cfg.paths.results_dir,
        target_facts=50,
        noise_tiers=[0, 200, 1000, 5000],
        seed=cfg.dataset.seed,
    )
    return bm.run()


def _run_conflict_quality(cfg, memories, queries):
    from suites.conflict_quality import ConflictQualityBenchmark
    bm = ConflictQualityBenchmark(
        db_path=_new_db(),
        output_dir=cfg.paths.results_dir,
        n_pairs=30,
        seed=cfg.dataset.seed,
    )
    return bm.run()


def _run_lean_skynet(cfg, memories, queries):
    from suites.lean_skynet import LeanSkynetBenchmark
    return LeanSkynetBenchmark(
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
    ).run()


def _run_baseline(cfg, memories, queries):
    from suites.baseline import BaselineComparisonBenchmark
    bm = BaselineComparisonBenchmark(
        db_path=_new_db(),
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
    )
    return bm.run()


def _run_graph_reasoning(cfg, memories, queries):
    from suites.graph_reasoning import GraphReasoningBenchmark
    return GraphReasoningBenchmark(
        db_path=_new_db(),
        output_dir=cfg.paths.results_dir,
        n_chains=30,
        seed=cfg.dataset.seed,
    ).run()


def _run_channel_ablation(cfg, memories, queries):
    from suites.channel_ablation import ChannelAblationBenchmark
    return ChannelAblationBenchmark(
        db_path=_new_db(),
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
    ).run()


def _run_hnsw_exactness(cfg, memories, queries):
    from suites.hnsw_exactness import HNSWExactnessBenchmark
    return HNSWExactnessBenchmark(
        memories=memories,
        queries=queries,
        output_dir=cfg.paths.results_dir,
        tiers=[1_000, 10_000],
    ).run()


def _run_dream_derived_fact(cfg, memories, queries):
    from suites.dream_derived_fact import DreamDerivedFactBenchmark
    # Codex v7 left "dream lift weak on real text (+0.04)" as the lone
    # remaining caveat. Scaling premises 25 -> 75 gives the Insight phase
    # more candidate clusters to materialise; k_strict 3 -> 5 widens the
    # strict-metric window so a derived:cluster in rank 4-5 also counts
    # (1 derived fact vs ~300 distractors at k=3 is a 1% population fit;
    # at k=5 it's 1.7%, still tight but a fairer test).
    return DreamDerivedFactBenchmark(
        db_path=_new_db(),
        output_dir=cfg.paths.results_dir,
        n_premises=75,
        k_strict=5,
        n_distractors=600,
        seed=cfg.dataset.seed,
    ).run()


def _run_continuity_controls(cfg, memories, queries):
    from suites.continuity_controls import ContinuityControlsBenchmark
    return ContinuityControlsBenchmark(
        db_path=_new_db(),
        output_dir=cfg.paths.results_dir,
        target_facts=50,
        noise_tiers=[0, 200, 1000, 5000],
        seed=cfg.dataset.seed,
    ).run()


SUITE_RUNNERS = {
    "retrieval":   _run_retrieval,
    "dream":       _run_dream,
    "gpu":         _run_gpu,
    "scalability": _run_scalability,
    "graph":       _run_graph,
    "concurrent":  _run_concurrent,
    "conflict":    _run_conflict,
    "agentic":     _run_agentic,
    "qa":          _run_qa,
    # v2 suites
    "diversity":          _run_diversity,
    "lstm_knn":           _run_lstm_knn,
    "continuity":         _run_continuity,
    "conflict_quality":   _run_conflict_quality,
    "baseline":           _run_baseline,
    # v3 suites — codex 2026-04-28
    "graph_reasoning":    _run_graph_reasoning,
    "channel_ablation":   _run_channel_ablation,
    "hnsw_exactness":     _run_hnsw_exactness,
    "dream_derived_fact": _run_dream_derived_fact,
    "continuity_controls": _run_continuity_controls,
    # v5 suites — codex 2026-04-28 caveat fixes:
    "lean_skynet":        _run_lean_skynet,
}


def run_suite(suite_name: str, cfg: BenchmarkConfig,
              memories: List[Dict], queries: List[Dict]) -> Dict[str, Any]:
    runner = SUITE_RUNNERS.get(suite_name)
    if runner is None:
        return {"error": f"Unknown suite: {suite_name}. Available: {list(SUITE_RUNNERS)}"}

    print(banner(f"Running suite: {suite_name.upper()}"))
    try:
        return runner(cfg, memories, queries)
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[-1000:]}


# ── Orchestrator ─────────────────────────────────────────────────────────────

class NeuralMemoryBenchmark:
    """Run all (or selected) suites, collect results, save to JSON."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.cfg = config or BenchmarkConfig()
        self.cfg.paths.ensure()
        self.start_time = time.perf_counter()
        self.results = {
            "meta": {
                "started_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "config": {
                    "memory_db": self.cfg.memory.db_path,
                    "suites": self.cfg.suites or "all",
                },
            },
            "suites": {},
            "errors": {},
        }

    def run(self) -> Dict[str, Any]:
        # Hot-path guard: refuse to start if any configured path resolves to the
        # production DB. Each suite uses _new_db() which double-checks too.
        _assert_not_hotpath(self.cfg.memory.db_path)

        print(banner("NEURAL MEMORY BENCHMARK SUITE"))
        print(f"Output: {self.cfg.paths.output_dir}")
        print(f"DB:     {self.cfg.memory.db_path}")
        print(f"Suites: {self.cfg.suites or 'ALL'}")
        print(f"Started: {datetime.now().isoformat()}")

        print(banner("Generating Dataset"))
        # Realistic mode: real prose from the project itself (.md + .py).
        # Codex v5 caveat — "synthetic data only" — addressed by
        # dataset_real.RealTextGenerator which produces (chunk, query)
        # pairs over actual repository text. Anchors are real
        # identifiers (snake_case / CamelCase / *.py) that occur in
        # exactly one chunk thanks to the global registry.
        if getattr(self.cfg, "realistic", False):
            from dataset_real import RealTextGenerator
            # Floor at 200 so the default queries_per_tier=50 doesn't cap
            # the real-text run (codex v6 caught that the prior pass was
            # only n=50, calling it a 'small slice' caveat).
            n = max(self.cfg.dataset.queries_per_tier or 0, 200)
            rgen = RealTextGenerator(seed=self.cfg.dataset.seed)
            memories, queries = rgen.generate(n)
            print(f"Generated {len(memories)} REAL-TEXT memories from project corpus")
            print(f"Generated {len(queries)} concept queries over real prose")
        # Paraphrase mode swaps the bag-of-tokens query generator for the
        # disjoint-vocabulary one in dataset_v2. The synthetic anchor
        # tokens make ground-truth a 1:1 lookup that can't be cheated by
        # lexical overlap, so the v2 suites (diversity / lstm_knn /
        # baseline / dream-recall-quality) have a clean signal.
        elif getattr(self.cfg, "paraphrase", False):
            from dataset_v2 import ParaphraseGenerator
            n = self.cfg.dataset.queries_per_tier or 200
            pgen = ParaphraseGenerator(seed=self.cfg.dataset.seed)
            memories, queries = pgen.generate(n)
            # Top up with classic memories so suites that need scale
            # (scalability, concurrent) still have a corpus to chew on.
            ds = MasterDataset(seed=self.cfg.dataset.seed)
            extras = ds.generate(
                episodic=self.cfg.dataset.episodic_count // 4,
                factual=self.cfg.dataset.factual_count // 4,
                temporal=0, conversational=0, graph=0, adversarial=0,
            )
            memories = memories + extras
            print(f"Generated {len(memories)} memories ({n} paraphrase + {len(extras)} extras)")
            print(f"Generated {len(queries)} paraphrase queries (1:1 ground truth, disjoint vocab)")
        else:
            ds = MasterDataset(seed=self.cfg.dataset.seed)
            memories = ds.generate(
                episodic=self.cfg.dataset.episodic_count,
                factual=self.cfg.dataset.factual_count,
                temporal=self.cfg.dataset.temporal_count,
                conversational=self.cfg.dataset.conversational_count,
                graph=self.cfg.dataset.graph_node_count,
                adversarial=self.cfg.dataset.adversarial_count,
            )
            print(f"Generated {len(memories)} memories")

            qgen = QueryGenerator(memories, seed=self.cfg.dataset.seed)
            queries = qgen.generate_recall_queries(count=self.cfg.dataset.queries_per_tier)
            print(f"Generated {len(queries)} recall queries")

        suites_to_run = self.cfg.suites if self.cfg.suites else ALL_SUITES

        for suite in suites_to_run:
            t0 = time.perf_counter()
            try:
                result = run_suite(suite, self.cfg, memories, queries)
                elapsed = time.perf_counter() - t0
                status = "error" if isinstance(result, dict) and "error" in result and len(result) <= 2 else "ok"
                self.results["suites"][suite] = {
                    "result": result,
                    "elapsed_s": round(elapsed, 2),
                    "status": status,
                }
                if status == "ok":
                    print(f"\n[OK] {suite} completed in {elapsed:.1f}s")
                else:
                    self.results["errors"][suite] = result.get("error", "unknown")
                    print(f"\n[ERROR] {suite}: {result.get('error', 'unknown')}")
            except Exception as e:
                import traceback
                elapsed = time.perf_counter() - t0
                self.results["suites"][suite] = {
                    "elapsed_s": round(elapsed, 2),
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()[-1000:],
                }
                self.results["errors"][suite] = str(e)
                print(f"\n[ERROR] {suite}: {e}")

        self.results["meta"]["finished_at"] = datetime.now().isoformat()
        self.results["meta"]["total_elapsed_s"] = round(
            time.perf_counter() - self.start_time, 2
        )

        self._save_results()
        return self.results

    def _save_results(self):
        out_path = self.cfg.paths.results_dir / "full_benchmark_results.json"
        out_path.write_text(json.dumps(self.results, indent=2, default=str))
        print(banner(f"Results saved to {out_path}"))


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Mazemaker Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.neural_memory_benchmark.runner
  python -m benchmarks.neural_memory_benchmark.runner --suite retrieval
  python -m benchmarks.neural_memory_benchmark.runner --suite retrieval --suite gpu
  python -m benchmarks.neural_memory_benchmark.runner --dry-run
  python -m benchmarks.neural_memory_benchmark.runner --output-dir /tmp/results
  python -m benchmarks.neural_memory_benchmark.runner --seed 0
  python -m benchmarks.neural_memory_benchmark.runner --list
        """,
    )
    parser.add_argument("--suite", action="append", dest="suites", metavar="NAME",
                        help="Suite to run (can be repeated)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results output directory")
    parser.add_argument("--db", dest="db_path", default=None,
                        help="Mazemaker DB path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate dataset only, don't run benchmarks")
    parser.add_argument("--list", action="store_true",
                        help="List available suites and exit")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Use the disjoint-vocabulary paraphrase dataset "
                             "(dataset_v2) for queries — required for the v2 "
                             "suites (diversity, lstm_knn, baseline, dream).")
    parser.add_argument("--realistic", action="store_true",
                        help="Use real prose from the project itself "
                             "(dataset_real) — addresses the codex v5 "
                             "'synthetic-only' caveat. Mutually exclusive "
                             "with --paraphrase; --realistic wins.")

    args = parser.parse_args()

    if args.list:
        print("Available suites:")
        for s in ALL_SUITES:
            print(f"  - {s}")
        return

    cfg = BenchmarkConfig.from_args()
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir
        cfg.paths.results_dir = args.output_dir / "results"
        cfg.paths.data_dir = args.output_dir / "data"
        cfg.paths.ensure()
    if args.db_path is not None:
        cfg.memory.db_path = args.db_path
    cfg.dataset.seed = args.seed
    cfg.suites = args.suites or []
    cfg.dry_run = args.dry_run
    cfg.paraphrase = bool(args.paraphrase)
    cfg.realistic = bool(args.realistic)

    if args.dry_run:
        print("Generating dataset only (dry run)...")
        ds = MasterDataset(seed=cfg.dataset.seed)
        memories = ds.generate(
            episodic=cfg.dataset.episodic_count,
            factual=cfg.dataset.factual_count,
            temporal=cfg.dataset.temporal_count,
            conversational=cfg.dataset.conversational_count,
            graph=cfg.dataset.graph_node_count,
            adversarial=cfg.dataset.adversarial_count,
        )
        print(f"Generated {len(memories)} memories")
        print("\nMemory breakdown:")
        from collections import Counter
        counts = Counter(m["label"].split(":")[0] for m in memories)
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return

    bm = NeuralMemoryBenchmark(cfg)
    bm.run()


if __name__ == "__main__":
    main()
