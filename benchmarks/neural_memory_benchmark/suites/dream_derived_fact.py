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
    Direct cosine + mazemaker recall both struggle because the
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

from memory_client import Mazemaker
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
    """LEGACY: True if the top-k results COLLECTIVELY contain both tokens.

    Marked legacy_collective_metric_inflated: with small corpora and k>=5 the
    union of top-k almost always contains P1 and P2 separately, so the metric
    saturates pre-dream and leaves no headroom for the dream engine to lift.
    Kept for backwards comparability only — read the strict metrics instead.
    """
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


def _single_doc_both_tokens(results: List[Dict[str, Any]], a_tok: str, b_tok: str) -> bool:
    """STRICT: True iff a SINGLE result contains BOTH tokens.

    Pre-dream this is 0 by construction — every premise template puts only
    a_tok in P1 and only b_tok in P2; no single original memory carries both.
    A hit therefore proves the dream engine synthesised a record (e.g. a
    derived:cluster) that fuses the two premises.
    """
    al = a_tok.lower()
    bl = b_tok.lower()
    for r in results:
        c = (r.get("content") or "").lower()
        if al in c and bl in c:
            return True
    return False


def _derived_cluster_in_topk(results: List[Dict[str, Any]]) -> bool:
    """STRICT: True iff any top-k result is a derived:* memory.

    The dream engine's Insight phase materialises memories with a
    'derived:cluster' (or other 'derived:') label. None exist pre-dream,
    so a hit here is the unambiguous signal that the dream engine
    produced AND surfaced a synthesised fact for the conjunction query.
    """
    for r in results:
        label = (r.get("label") or r.get("metadata", {}).get("label") or "")
        if isinstance(label, str) and label.startswith("derived:"):
            return True
    return False


def _measure(nm: Mazemaker, queries: List[Dict[str, Any]], k: int = 5,
             k_strict: int = 3) -> Dict[str, Any]:
    multihop_both = 0
    semantic_both = 0
    # Strict metrics use a tighter top-k (default 3) to remove the collective
    # top-5 inflation Codex flagged.
    sem_single_doc = 0
    mh_single_doc = 0
    sem_derived = 0
    mh_derived = 0
    n = len(queries) or 1
    # F65 fix (audit 2026-05-13): the silent `except Exception` masked
    # both "API not implemented" and "API broken" — both got 0 hits and
    # the report blamed the engine. Distinguish: probe once up-front;
    # if missing, record availability=False so the report can interpret
    # the numbers correctly.
    multihop_available = hasattr(nm, "recall_multihop")
    multihop_first_error: Optional[str] = None
    for q in queries:
        if multihop_available:
            try:
                mh = nm.recall_multihop(q["query"], k=k, hops=2)
            except Exception as e:
                mh = []
                if multihop_first_error is None:
                    multihop_first_error = f"{type(e).__name__}: {e}"
        else:
            mh = []
        if _both_tokens_present(mh, q["a_tok"], q["b_tok"]):
            multihop_both += 1
        sem = nm.recall(q["query"], k=k)
        if _both_tokens_present(sem, q["a_tok"], q["b_tok"]):
            semantic_both += 1

        # Strict pass — re-query at k_strict so collective-top-k inflation
        # cannot mask a real lift. recall() is cheap; re-issuing keeps the
        # legacy numbers stable for downstream regressions.
        try:
            mh_strict = nm.recall_multihop(q["query"], k=k_strict, hops=2)
        except Exception:
            mh_strict = []
        sem_strict = nm.recall(q["query"], k=k_strict)
        if _single_doc_both_tokens(sem_strict, q["a_tok"], q["b_tok"]):
            sem_single_doc += 1
        if _single_doc_both_tokens(mh_strict, q["a_tok"], q["b_tok"]):
            mh_single_doc += 1
        if _derived_cluster_in_topk(sem_strict):
            sem_derived += 1
        if _derived_cluster_in_topk(mh_strict):
            mh_derived += 1
    return {
        # Legacy — collective top-k saturation; do not interpret without context.
        "legacy_collective_metric_inflated": True,
        "semantic_both_tokens_rate": round(semantic_both / n, 4),
        "multihop_both_tokens_rate": round(multihop_both / n, 4),
        # Strict metrics — only fireable by a dream-synthesised record.
        "single_doc_both_tokens_rate_semantic": round(sem_single_doc / n, 4),
        "single_doc_both_tokens_rate_multihop": round(mh_single_doc / n, 4),
        "derived_fact_hit_rate_semantic": round(sem_derived / n, 4),
        "derived_fact_hit_rate_multihop": round(mh_derived / n, 4),
        "k_legacy": k,
        "k_strict": k_strict,
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
        k_strict: int = 3,
        n_distractors: int = 300,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.n_premises = n_premises
        self.seed = seed
        self.k = k
        self.k_strict = k_strict
        self.n_distractors = n_distractors
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Dream Derived-Fact Benchmark ===")
        memories, queries = _build_premises(self.n_premises, self.seed)
        print(f"  Built {self.n_premises} premise-pairs ({len(memories)} memories, "
              f"{len(queries)} conjunction queries)")

        nm = Mazemaker(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )

        # Inject paraphrase distractors BEFORE premises so the conjunction
        # queries have to discriminate the right entity from a noisy corpus.
        # Distractors land under a 'distractor:' label so they are excluded
        # from the premise-count metric and easy to filter in analysis.
        #
        # F66 fix (audit 2026-05-13): distractors used to ingest with
        # `auto_connect=True`, creating uncontrolled random edges to
        # the premise memories and contaminating the graph the dream
        # engine then operates on. Disable auto-connect for distractors
        # so the graph the engine sees is the controlled premise graph
        # only — distractors influence semantic recall but not graph
        # traversal.
        if self.n_distractors > 0:
            pg = ParaphraseGenerator(seed=self.seed + 9001)
            distractor_mems, _ = pg.generate(self.n_distractors)
            for dm in distractor_mems:
                nm.remember(
                    dm["text"],
                    label=f"distractor:{dm.get('label', 'paraphrase')}",
                    auto_connect=False,
                )
            print(f"  Injected {self.n_distractors} paraphrase distractors (no auto-connect)")

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
                "recall": _measure(nm, queries, self.k, self.k_strict),
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
            "recall": _measure(nm, queries, self.k, self.k_strict),
        }
        print(f"  post-dream: {results['post_dream']['recall']}  "
              f"connections={post_conns} derived={post_derived}  "
              f"dream={dream_s:.2f}s")

        pre_r = results["pre_dream"]["recall"]
        post_r = results["post_dream"]["recall"]
        results["lift"] = {
            # Legacy lifts — kept for comparability, but inflated.
            "semantic_both_tokens_legacy": round(
                post_r["semantic_both_tokens_rate"]
                - pre_r["semantic_both_tokens_rate"], 4),
            "multihop_both_tokens_legacy": round(
                post_r["multihop_both_tokens_rate"]
                - pre_r["multihop_both_tokens_rate"], 4),
            # Strict lifts — these are the ones to read.
            "single_doc_both_tokens_semantic": round(
                post_r["single_doc_both_tokens_rate_semantic"]
                - pre_r["single_doc_both_tokens_rate_semantic"], 4),
            "single_doc_both_tokens_multihop": round(
                post_r["single_doc_both_tokens_rate_multihop"]
                - pre_r["single_doc_both_tokens_rate_multihop"], 4),
            "derived_fact_hit_rate_semantic": round(
                post_r["derived_fact_hit_rate_semantic"]
                - pre_r["derived_fact_hit_rate_semantic"], 4),
            "derived_fact_hit_rate_multihop": round(
                post_r["derived_fact_hit_rate_multihop"]
                - pre_r["derived_fact_hit_rate_multihop"], 4),
            "new_connections": post_conns - pre_conns,
            "new_derived_facts": post_derived - pre_derived,
            "interpretation": (
                "Read the STRICT metrics (single_doc_both_tokens, "
                "derived_fact_hit_rate). Pre-dream both must be 0 by "
                "construction: no premise template puts both a_tok and "
                "b_tok in the same memory and no derived:* labels exist. "
                "Post-dream derived_fact_hit_rate > 0 is the unambiguous "
                "signal that the Insight phase materialised a synthesised "
                "memory AND retrieval surfaced it. The legacy "
                "*_both_tokens_rate fields use a collective top-k metric "
                "and saturate above ~0.9 even pre-dream — do not interpret "
                "them as evidence either way (legacy_collective_metric_inflated)."
            ),
        }
        print(f"  lift: {results['lift']}")

        out = self.output_dir / "dream_derived_fact_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
