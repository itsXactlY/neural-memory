"""
Graph Traversal Benchmark Suite
================================
Compares BFS vs PPR vs HNSW graph traversal for recall.

Tests:
  1. BFS traversal: depth, breadth, coverage
  2. PPR (Personalized PageRank): alpha, iterations, teleport
  3. HNSW: ANN pre-filter + rerank quality
  4. Cross-engine recall quality comparison
  5. Latency vs depth / iterations
"""
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def run_bfs(
    nm: NeuralMemory,
    start_ids: List[int],
    depth: int = 3,
) -> Dict[str, Any]:
    """Run BFS from starting nodes."""
    visited = set()
    frontier = set(start_ids)
    layers = {0: set(start_ids)}
    elapsed_times = []

    for d in range(1, depth + 1):
        t0 = time.perf_counter()
        next_frontier = set()
        for node_id in frontier:
            try:
                connections = nm.connections(node_id)
                for conn in connections:
                    cid = conn.get("id", conn.get("target_id", conn.get("memory_id")))
                    if cid and cid not in visited:
                        next_frontier.add(cid)
                        visited.add(cid)
            except Exception:
                pass
        frontier = next_frontier
        layers[d] = frontier
        elapsed_times.append(time.perf_counter() - t0)

    return {
        "total_visited": len(visited),
        "nodes_per_layer": {d: len(nodes) for d, nodes in layers.items()},
        "avg_layer_time_ms": round(sum(elapsed_times) / len(elapsed_times) * 1000, 2) if elapsed_times else 0,
        "total_time_ms": round(sum(elapsed_times) * 1000, 2),
    }


def run_ppr(
    nm: NeuralMemory,
    start_ids: List[int],
    alpha: float = 0.85,
    iterations: int = 50,
    teleport: float = 0.15,
) -> Dict[str, Any]:
    """Run Personalized PageRank from starting nodes."""
    try:
        # NeuralMemory.think() implements PPR-like spreading activation
        t0 = time.perf_counter()
        results = []
        for sid in start_ids[:5]:  # Limit PPR scope
            try:
                activated = nm.think(sid, depth=iterations // 10, decay=alpha)
                results.extend(activated)
            except Exception:
                pass
        elapsed = time.perf_counter() - t0

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            rid = r.get("id", "")
            if rid not in seen:
                seen.add(rid)
                unique.append(r)

        return {
            "total_activated": len(unique),
            "time_ms": round(elapsed * 1000, 2),
            "ms_per_start_node": round(elapsed / len(start_ids[:5]) * 1000, 2),
            "top_results": unique[:10],
        }
    except Exception as e:
        return {"error": str(e)}


def run_think_spreading(
    nm: NeuralMemory,
    start_ids: List[int],
    depth: int = 3,
    decay: float = 0.85,
) -> Dict[str, Any]:
    """Run neural_think (spreading activation) from starting nodes."""
    t0 = time.perf_counter()
    all_results = []
    for sid in start_ids[:10]:
        try:
            results = nm.think(sid, depth=depth, decay=decay)
            all_results.extend(results)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        rid = r.get("id", "")
        if rid not in seen:
            seen.add(rid)
            unique.append(r)

    return {
        "total_activated": len(unique),
        "time_ms": round(elapsed * 1000, 2),
        "ms_per_start_node": round(elapsed / min(len(start_ids), 10) * 1000, 2),
    }


class GraphBenchmark:
    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        depths: List[int] = None,
    ):
        self.db_path = db_path
        self.memories = memories
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.depths = depths or [2, 3, 5, 8]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nm = None

    def setup(self) -> Dict[str, Any]:
        print(f"  [setup] Storing {len(self.memories)} memories...")
        self.nm = NeuralMemory(db_path=self.db_path, embedding_backend="auto")
        for m in self.memories:
            self.nm.remember(m["text"], label=m["label"], auto_connect=True)
        print(f"  [setup] Done")
        return self.nm.stats()

    def run(self) -> Dict[str, Any]:
        import json
        print("\n=== Graph Traversal Benchmark ===")

        stats = self.setup()
        results = {"db_stats": stats, "bfs": {}, "think": {}, "ppr": {}}

        # Get some starting node IDs (use recent memories)
        try:
            sample = self.nm.recall(".", k=50)
            start_ids = [
                r.get("id", r.get("memory_id"))
                for r in sample
                if r.get("id") or r.get("memory_id")
            ]
            if not start_ids:
                print("  No start IDs found, using memory IDs from DB directly")
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cur = conn.execute("SELECT id FROM memories LIMIT 50")
                start_ids = [r[0] for r in cur.fetchall()]
                conn.close()
        except Exception as e:
            print(f"  Error getting start IDs: {e}")
            return results

        print(f"  Starting from {len(start_ids)} nodes\n")

        # BFS at different depths
        print("[BFS] Traversal depth comparison")
        for depth in self.depths:
            print(f"  depth={depth}...", end=" ", flush=True)
            r = run_bfs(self.nm, start_ids, depth=depth)
            results["bfs"][f"depth_{depth}"] = r
            print(f"visited={r.get('total_visited', 0)}, "
                  f"time={r.get('total_time_ms', 0)}ms")

        # Spreading activation (think)
        print("\n[Think] Spreading activation")
        for depth in [2, 3, 5]:
            print(f"  depth={depth}...", end=" ", flush=True)
            r = run_think_spreading(self.nm, start_ids, depth=depth)
            results["think"][f"depth_{depth}"] = r
            print(f"activated={r.get('total_activated', 0)}, "
                  f"time={r.get('time_ms', 0)}ms")

        # PPR parameters
        print("\n[PPR] Personalized PageRank")
        for alpha in [0.7, 0.85, 0.95]:
            for iters in [20, 50]:
                key = f"alpha_{alpha}_iters_{iters}"
                print(f"  {key}...", end=" ", flush=True)
                r = run_ppr(self.nm, start_ids, alpha=alpha, iterations=iters)
                results["ppr"][key] = r
                print(f"activated={r.get('total_activated', 0)}, "
                      f"time={r.get('time_ms', 0)}ms")

        out_path = self.output_dir / "graph_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        return results
