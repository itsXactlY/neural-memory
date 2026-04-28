"""
Dream Derived-Fact Benchmark
=============================
The dream engine's claim is that REM creates bridges between
isolated memories and Insight materialises new derived:cluster
memories. This suite tests whether ANY of that machinery actually
helps recall — by querying for facts that are STRUCTURALLY only
inferable post-dream.

Construction:
  * Build "premise pairs": memory P1 about entity X with attribute A,
    memory P2 about entity X with attribute B. NEITHER P1 nor P2
    contains the synthesis "X has both A and B"; that conjunction
    only exists if the dream engine derives a cluster memory for X
    spanning P1 and P2.
  * Pre-dream query for the conjunction: "List both attributes of X."
    Direct cosine + neural-memory recall both struggle because the
    answer is split across two records.
  * Run DreamEngine.dream_now(). If REM/Insight materialise a
    derived:cluster memory for X (or strengthen the P1<->P2 edge so
    multihop can stitch them), recall should improve.
  * Re-query and report the post-dream lift.

This is the only suite where the dream engine's value is measured by
RECALL improvement, not just structural deltas (edge counts, derived
memory counts).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory
from dream_engine import DreamEngine

try:
    from dataset_v2 import ParaphraseGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import ParaphraseGenerator


_PREMISE_TEMPLATES = [
    # (P1 template, P2 template, query template, expected_a_token, expected_b_token)
    (
        "Component {x} runs on the colocated rack.",
        "Component {x} is scheduled to migrate to a hyperscaler region.",
        "Where does component {x} live now and where will it move?",
        "rack",
        "hyperscaler",
    ),
    (
        "Pipeline {x} is owned by squad alpha-cell.",
        "Pipeline {x} reports its SLOs through squad observability.",
        "Which squads touch pipeline {x}?",
        "alpha-cell",
        "observability",
    ),
    (
        "Service {x} stores transactional data in the relational tier.",
        "Service {x} archives cold rows to the columnar tier.",
        "How is data for service {x} split across stores?",
        "relational",
        "columnar",
    ),
    (
        "Job {x} runs nightly during the backup window.",
        "Job {x} also fires manually before every release.",
        "What schedules trigger job {x}?",
        "nightly",
        "release",
    ),
]


def _build_premises(n: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pg = ParaphraseGenerator(seed=seed)
    import random
    rng = random.Random(seed)
    memories: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    for _ in range(n):
        x = pg._fresh_anchor()
        p1, p2, qt, a_tok, b_tok = rng.choice(_PREMISE_TEMPLATES)
        memories.append({
            "text": p1.format(x=x),
            "label": f"premise:p1:{x}",
            "anchor": x,
        })
        memories.append({
            "text": p2.format(x=x),
            "label": f"premise:p2:{x}",
            "anchor": x,
        })
        queries.append({
            "query": qt.format(x=x),
            "anchor": x,
            "expects_both_tokens": True,
            "a_tok": a_tok,
            "b_tok": b_tok,
        })
    return memories, queries


def _both_tokens_present(results: List[Dict[str, Any]], a_tok: str, b_tok: str) -> bool:
    """True if the top-k results collectively contain both tokens."""
    al = a_tok.lower()
    bl = b_tok.lower()
    saw_a = saw_b = False
    for r in results:
        c = (r.get("content") or "").lower()
        if al in c:
            saw_a = True
        if bl in c:
            saw_b = True
        if saw_a and saw_b:
            return True
    return False


def _measure(nm: NeuralMemory, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    multihop_both = 0
    semantic_both = 0
    n = len(queries) or 1
    for q in queries:
        # Try multihop first — graph-aware.
        try:
            mh = nm.recall_multihop(q["query"], k=k, hops=2)
        except Exception:
            mh = []
        if _both_tokens_present(mh, q["a_tok"], q["b_tok"]):
            multihop_both += 1
        sem = nm.recall(q["query"], k=k)
        if _both_tokens_present(sem, q["a_tok"], q["b_tok"]):
            semantic_both += 1
    return {
        "semantic_both_tokens_rate": round(semantic_both / n, 4),
        "multihop_both_tokens_rate": round(multihop_both / n, 4),
        "n": n,
    }


class DreamDerivedFactBenchmark:
    def __init__(
        self,
        db_path: str,
        output_dir: Optional[Path] = None,
        n_premises: int = 25,
        seed: int = 42,
        k: int = 5,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.n_premises = n_premises
        self.seed = seed
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Dream Derived-Fact Benchmark ===")
        memories, queries = _build_premises(self.n_premises, self.seed)
        print(f"  Built {self.n_premises} premise-pairs ({len(memories)} memories, "
              f"{len(queries)} conjunction queries)")

        nm = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        for m in memories:
            nm.remember(m["text"], label=m["label"], auto_connect=True)

        # Snapshot derived:cluster count + connection count BEFORE dreaming.
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        pre_conns = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        pre_derived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE label LIKE 'derived:%'"
        ).fetchone()[0]
        conn.close()

        results: Dict[str, Any] = {
            "pre_dream": {
                "connections": pre_conns,
                "derived_facts": pre_derived,
                "recall": _measure(nm, queries, self.k),
            }
        }
        print(f"  pre-dream: {results['pre_dream']['recall']}  "
              f"connections={pre_conns} derived={pre_derived}")

        # Run a real dream cycle.
        engine = DreamEngine.sqlite(
            self.db_path, neural_memory=nm,
            idle_threshold=10**9, memory_threshold=10**9,
        )
        t0 = time.perf_counter()
        engine._phase_nrem()
        engine._phase_rem()
        engine._phase_insights()
        dream_s = time.perf_counter() - t0

        conn = sqlite3.connect(self.db_path)
        post_conns = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        post_derived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE label LIKE 'derived:%'"
        ).fetchone()[0]
        conn.close()

        results["post_dream"] = {
            "connections": post_conns,
            "derived_facts": post_derived,
            "dream_elapsed_s": round(dream_s, 2),
            "recall": _measure(nm, queries, self.k),
        }
        print(f"  post-dream: {results['post_dream']['recall']}  "
              f"connections={post_conns} derived={post_derived}  "
              f"dream={dream_s:.2f}s")

        results["lift"] = {
            "semantic_both_tokens": round(
                results["post_dream"]["recall"]["semantic_both_tokens_rate"]
                - results["pre_dream"]["recall"]["semantic_both_tokens_rate"],
                4,
            ),
            "multihop_both_tokens": round(
                results["post_dream"]["recall"]["multihop_both_tokens_rate"]
                - results["pre_dream"]["recall"]["multihop_both_tokens_rate"],
                4,
            ),
            "new_connections": post_conns - pre_conns,
            "new_derived_facts": post_derived - pre_derived,
            "interpretation": (
                "If lift.both_tokens > 0, the dream engine produced edges or "
                "derived facts that improved cross-memory recall. If both "
                "lifts are 0 even though new_connections/new_derived_facts > 0, "
                "the structural changes did not improve task quality."
            ),
        }
        print(f"  lift: {results['lift']}")

        out = self.output_dir / "dream_derived_fact_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
