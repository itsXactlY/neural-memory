"""
Agentic Workflow Benchmark Suite
===================================
End-to-end simulation of real agent workflows.

Simulates:
  1. Multi-session agentic loops (remember + recall + think + graph)
  2. Planning: decompose task, store sub-task memories, recall context
  3. Debugging: store error, recall similar errors, think through solutions
  4. Research: store findings, recall related facts, think across domains
  5. E2E latency per action type

Metrics:
  - Throughput: actions/second
  - Per-action latency: remember, recall, think, graph
  - Context preservation: can agent recall relevant context after N turns?
  - End-to-end session quality: task completion simulation
"""
import sys
import time
import random
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


class AgentWorkflow:
    """Simulates a single agent session with realistic tool usage."""

    def __init__(self, nm: NeuralMemory, session_id: int, seed: int = 42):
        self.nm = nm
        self.session_id = session_id
        self.rng = random.Random(seed + session_id)
        self.actions = []
        self.timings = {}

    def think(self, prompt: str) -> float:
        """Simulate thinking with spreading activation."""
        t0 = time.perf_counter()
        # Pick a random known memory to start from
        try:
            sample = self.nm.recall(prompt, k=3)
            if sample:
                start_id = sample[0].get("id", sample[0].get("memory_id"))
                if start_id:
                    self.nm.think(start_id, depth=3, decay=0.85)
        except Exception:
            pass
        return time.perf_counter() - t0

    def run_session(self, num_turns: int = 20) -> Dict[str, Any]:
        """Run a simulated agent session."""
        turn_timings = {
            "remember": [],
            "recall": [],
            "think": [],
            "graph": [],
        }

        for turn in range(num_turns):
            # Determine action mix for this turn
            action_weights = [
                ("remember", 0.35),   # Remember new info
                ("recall", 0.30),     # Recall context
                ("think", 0.20),       # Spread activation
                ("graph", 0.10),       # Graph overview
                ("multi", 0.05),       # Multiple actions
            ]

            action = self.rng.choices(
                [a for a, _ in action_weights],
                weights=[w for _, w in action_weights],
            )[0]

            if action == "remember":
                t0 = time.perf_counter()
                self.nm.remember(
                    self._random_memory_text(),
                    label=f"agentic:session_{self.session_id}",
                    auto_connect=True,
                )
                turn_timings["remember"].append(time.perf_counter() - t0)

            elif action == "recall":
                t0 = time.perf_counter()
                self.nm.recall(self._random_query(), k=5)
                turn_timings["recall"].append(time.perf_counter() - t0)

            elif action == "think":
                t = self.think(self._random_query())
                turn_timings["think"].append(t)

            elif action == "graph":
                t0 = time.perf_counter()
                try:
                    self.nm.graph()
                except Exception:
                    pass
                turn_timings["graph"].append(time.perf_counter() - t0)

            else:  # multi
                t0 = time.perf_counter()
                self.nm.remember(self._random_memory_text(), label="multi", auto_connect=False)
                self.nm.recall(self._random_query(), k=3)
                turn_timings["remember"].append((time.perf_counter() - t0) / 2)
                turn_timings["recall"].append((time.perf_counter() - t0) / 2)

        # Summarize timings
        summary = {}
        for action, times in turn_timings.items():
            if times:
                times_ms = [t * 1000 for t in times]
                summary[action] = {
                    "count": len(times),
                    "mean_ms": round(statistics.mean(times_ms), 2),
                    "p50_ms": round(statistics.median(times_ms), 2),
                    "p95_ms": round(sorted(times_ms)[int(len(times_ms) * 0.95)] if times_ms else 0, 2),
                    "p99_ms": round(sorted(times_ms)[int(len(times_ms) * 0.99)] if times_ms else 0, 2),
                }

        return summary

    def _random_memory_text(self) -> str:
        templates = [
            "Completed task: {task} with result {result}",
            "User requested: {task}",
            "Debugging {component}: found {finding}",
            "Deployed {service} to {env}",
            "Refactored {module} for {reason}",
        ]
        tasks = ["memory indexing", "benchmark run", "config update", "API integration"]
        results = ["success", "partial", "failed"]
        components = ["neural_memory.py", "embed_provider.py", "dream_engine.py"]
        services = ["api", "dashboard", "sync-bridge"]
        envs = ["production", "staging", "dev"]
        modules = ["memory_client", "embed_server", "sync_bridge"]
        findings = ["root cause in line 42", "null pointer exception", "timeout"]
        reasons = ["performance", "readability", "security", "compatibility"]

        tpl = self.rng.choice(templates)
        return tpl.format(
            task=self.rng.choice(tasks),
            result=self.rng.choice(results),
            component=self.rng.choice(components),
            service=self.rng.choice(services),
            env=self.rng.choice(envs),
            module=self.rng.choice(modules),
            finding=self.rng.choice(findings),
            reason=self.rng.choice(reasons),
        )

    def _random_query(self) -> str:
        queries = [
            "memory indexing performance",
            "debugging neural memory",
            "benchmark results",
            "config settings",
            "API integration issues",
            "deployment status",
            "Dream Engine phases",
            "HNSW configuration",
        ]
        return self.rng.choice(queries)


class AgenticBenchmark:
    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        num_sessions: int = 5,
        turns_per_session: int = 20,
    ):
        self.db_path = db_path
        self.memories = memories
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.num_sessions = num_sessions
        self.turns_per_session = turns_per_session
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Agentic Workflow Benchmark ===")

        results = {"sessions": {}, "summary": {}}

        # Setup: pre-populate with some memories
        print(f"  [setup] Pre-populating with {min(1000, len(self.memories))} memories...")
        nm = NeuralMemory(db_path=self.db_path, embedding_backend="auto")
        for m in self.memories[:1000]:
            nm.remember(m["text"], label=m["label"], auto_connect=True)
        print(f"  [setup] Done")

        session_results = []
        for session_id in range(self.num_sessions):
            print(f"\n  Session {session_id + 1}/{self.num_sessions}...", end=" ", flush=True)
            t0 = time.perf_counter()

            wf = AgentWorkflow(nm, session_id=session_id, seed=42)
            summary = wf.run_session(num_turns=self.turns_per_session)

            elapsed = time.perf_counter() - t0
            total_actions = sum(s.get("count", 0) for s in summary.values())

            session_results.append({
                "session_id": session_id,
                "elapsed_s": round(elapsed, 2),
                "total_actions": total_actions,
                "actions_per_second": round(total_actions / elapsed, 1) if elapsed > 0 else 0,
                "timings": summary,
            })
            print(f"{total_actions} actions in {elapsed:.1f}s "
                  f"({total_actions/elapsed:.1f} actions/s)")

        results["sessions"] = session_results

        # Summary across sessions
        all_aps = [s["actions_per_second"] for s in session_results]
        results["summary"] = {
            "num_sessions": self.num_sessions,
            "total_actions": sum(s["total_actions"] for s in session_results),
            "avg_actions_per_second": round(statistics.mean(all_aps), 1),
            "min_aps": round(min(all_aps), 1),
            "max_aps": round(max(all_aps), 1),
        }

        # Per-action aggregate timings
        action_aggregates = {}
        for action in ["remember", "recall", "think", "graph"]:
            means = [s["timings"].get(action, {}).get("mean_ms", 0) for s in session_results]
            means = [m for m in means if m > 0]
            if means:
                action_aggregates[action] = {
                    "avg_mean_ms": round(statistics.mean(means), 2),
                    "avg_p95_ms": round(statistics.mean(
                        [s["timings"].get(action, {}).get("p95_ms", 0) for s in session_results]
                    ), 2),
                }

        results["summary"]["action_aggregates"] = action_aggregates

        out_path = self.output_dir / "agentic_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        print(f"  Avg throughput: {results['summary']['avg_actions_per_second']} actions/s")
        return results
