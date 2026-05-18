"""
Graph Reasoning Benchmark — multi-hop semantic memory
=======================================================
The previous benchmark suite has been measuring "find the one document
that mentions this unique anchor." Raw cosine wins that trivially. This
suite tests something only a graph-augmented system *can* do:

  Build A -> B -> C chains where the query mentions A but the answer
  text lives only on C.  Direct cosine similarity from the query to C
  is low — the only way to find C is to retrieve B (semantically close
  to A's mention) and then traverse the edge B->C.

Each chain is constructed so that:
  * The QUERY embedding is closest to A (anchor + topic).
  * A's MEMORY says "see also B" using B's anchor.
  * B's MEMORY says "see also C" using C's anchor.
  * C's MEMORY contains the actual answer token.
  * No memory other than C contains the answer token.
  * Edges are added EXPLICITLY (auto_connect=False) — exactly two per
    chain (A→B and B→C, no A→C). Otherwise auto_connect's cosine
    threshold either over-links (creating an A↔C shortcut that
    trivialises multihop) or under-links (leaving the chain
    disconnected so the multihop test is unwinnable). Either way the
    suite stops measuring traversal — that was the GPT-5.5 codex
    finding that motivated this rewrite.

We measure hop-distance recall:
  * Direct cosine (numpy baseline) — should fail on hop-2 chains.
  * Mazemaker.recall() — should also fail without traversal.
  * Mazemaker.recall_multihop() — should hit hop-2 via PPR/think.
  * Mazemaker.think() from the anchor — should reach C.

Plus a control: queries with shuffled chain edges (graph broken) — the
multihop pipeline should DROP back to baseline, proving the lift was
graph-driven not semantic-driven.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import Mazemaker

try:
    from dataset_v2 import ParaphraseGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import ParaphraseGenerator


def _build_chain(anchor_a: str, anchor_b: str, anchor_c: str, answer: str) -> List[Dict[str, Any]]:
    """Three memories forming an A -> B -> C reasoning chain.

    A's text mentions B by anchor (forces edge to B during auto_connect's
    cosine threshold).  B's text mentions C.  C's text contains the answer
    token. Only C contains the answer.
    """
    return [
        {
            "id": f"chain-A-{anchor_a}",
            "anchor": anchor_a,
            "role": "A",
            "text": (
                f"Subsystem {anchor_a} routes its decisions through "
                f"the {anchor_b} planner. Refer there for outcomes."
            ),
            "label": f"chain:A:{anchor_a}",
        },
        {
            "id": f"chain-B-{anchor_b}",
            "anchor": anchor_b,
            "role": "B",
            "text": (
                f"The {anchor_b} planner forwards every approved request "
                f"to the {anchor_c} executor downstream."
            ),
            "label": f"chain:B:{anchor_b}",
        },
        {
            "id": f"chain-C-{anchor_c}",
            "anchor": anchor_c,
            "role": "C",
            "text": (
                f"The {anchor_c} executor reports its final disposition "
                f"as {answer}."
            ),
            "label": f"chain:C:{anchor_c}",
        },
    ]


def _build_chains(n: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate n A->B->C chains plus paraphrased queries about A->C."""
    pg = ParaphraseGenerator(seed=seed)
    rng = random.Random(seed)
    answers = [
        "ratified", "rolled-back", "queued for review", "auto-merged",
        "blocked by policy", "deferred to next cycle", "redirected to fallback",
        "shadow-published",
    ]
    memories: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    for _ in range(n):
        a = pg._fresh_anchor()
        b = pg._fresh_anchor()
        c = pg._fresh_anchor()
        ans = rng.choice(answers)
        chain = _build_chain(a, b, c, ans)
        memories.extend(chain)
        # Query mentions ONLY A and asks about A's outcome — answer is on C.
        queries.append({
            "query": f"What is the disposition of subsystem {a}?",
            "anchor_a": a,
            "anchor_c": c,
            "answer_token": ans,
            "ground_truth_id": f"chain-C-{c}",  # C is the only correct hit
        })
    return memories, queries


def _rank_of_anchor(anchor: str, results: List[Dict[str, Any]]) -> int:
    for i, r in enumerate(results, 1):
        if anchor and anchor in (r.get("content") or ""):
            return i
    return 0


def _measure_pipeline(
    label: str,
    recall_fn,
    queries: List[Dict[str, Any]],
    k: int = 10,
) -> Dict[str, Any]:
    hits_C = 0
    hits_C_top3 = 0
    rrs: List[float] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = recall_fn(q["query"], k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        # The answer (C) is what matters; A and B reaching the top-k means the
        # pipeline got the LOCAL hit but didn't traverse — partial credit.
        rank_c = _rank_of_anchor(q["anchor_c"], results)
        if rank_c > 0:
            hits_C += 1
        if 0 < rank_c <= 3:
            hits_C_top3 += 1
        rrs.append(1.0 / rank_c if rank_c > 0 else 0.0)
    n = len(queries) or 1
    latencies_ms.sort()
    return {
        "label": label,
        "recall_C_at_k": round(hits_C / n, 4),
        "recall_C_at_3": round(hits_C_top3 / n, 4),
        "mrr_C": round(statistics.mean(rrs), 4),
        "p50_ms": round(latencies_ms[len(latencies_ms) // 2], 3),
        "p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95)], 3),
        "n": n,
    }


class GraphReasoningBenchmark:
    def __init__(
        self,
        db_path: str,
        output_dir: Optional[Path] = None,
        n_chains: int = 30,
        k: int = 10,
        seed: int = 42,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.n_chains = n_chains
        self.k = k
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Graph Reasoning Benchmark (A→B→C chains) ===")
        memories, queries = _build_chains(self.n_chains, self.seed)
        print(f"  Built {self.n_chains} chains "
              f"({len(memories)} memories, {len(queries)} hop-2 queries)")

        # Two raw-cosine sanity checks:
        #   * cosine over chain memories alone — measures whether a vanilla
        #     vector store can find C from a query about A.
        # We import locally to keep the suite self-contained.
        import numpy as np
        from embed_provider import EmbeddingProvider
        emb = EmbeddingProvider(backend="auto")

        chain_embs = np.asarray(
            [emb.embed(m["text"]) for m in memories], dtype=np.float32
        )
        chain_embs /= np.linalg.norm(chain_embs, axis=1, keepdims=True).clip(min=1e-12)

        def raw_recall(q: str, k: int = 10) -> List[Dict[str, Any]]:
            qv = np.asarray(emb.embed(q), dtype=np.float32)
            qv /= np.linalg.norm(qv).clip(min=1e-12)
            sims = chain_embs @ qv
            idxs = np.argsort(-sims)[:k]
            return [
                {
                    "id": memories[i]["id"],
                    "label": memories[i]["label"],
                    "content": memories[i]["text"],
                    "similarity": float(sims[i]),
                }
                for i in idxs
            ]

        # Mazemaker side. Critically, auto_connect is DISABLED — auto_connect
        # creates edges by cosine similarity of the chain texts, which both
        # (a) sometimes spuriously links A↔C directly (because they share the
        # chain's topical vocabulary), trivialising multihop, and (b) sometimes
        # fails to link A→B / B→C, leaving the chain disconnected. Either way
        # the suite stops actually testing graph traversal. The fix: ingest with
        # auto_connect=False, then EXPLICITLY add A→B and B→C edges (and only
        # those) so the only way from A to C is via a real graph hop.
        nm = Mazemaker(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        # Map per-chain (a, b, c anchors) → (a_id, b_id, c_id) DB ids.
        chain_ids: List[Tuple[int, int, int]] = []
        memory_ids: List[int] = []
        for i in range(0, len(memories), 3):
            a_mem, b_mem, c_mem = memories[i], memories[i + 1], memories[i + 2]
            a_id = int(nm.remember(a_mem["text"], label=a_mem["label"], auto_connect=False))
            b_id = int(nm.remember(b_mem["text"], label=b_mem["label"], auto_connect=False))
            c_id = int(nm.remember(c_mem["text"], label=c_mem["label"], auto_connect=False))
            chain_ids.append((a_id, b_id, c_id))
            memory_ids.extend([a_id, b_id, c_id])

        # Explicit chain edges. add_connection canonicalises source<target,
        # so we don't care which way we pass the args — the canonical row is
        # what survives. We use edge_type='references' (matches A's "see also B"
        # semantics) and weight 0.7 (above any reasonable similarity floor).
        for a_id, b_id, c_id in chain_ids:
            nm.store.add_connection(a_id, b_id, weight=0.7, edge_type="references")
            nm.store.add_connection(b_id, c_id, weight=0.7, edge_type="references")

        # ------------------------------------------------------------------
        # Verify the graph is what we claim it is. These assertions catch the
        # exact regression Codex found: auto_connect creating A↔C shortcuts
        # or skipping chain edges entirely.
        # ------------------------------------------------------------------
        # F73 fix (audit 2026-05-13): the previous code did
        # `fetchone()["n"]` which assumes Row factory is set; otherwise
        # it raises TypeError. Index by position instead — robust regardless
        # of the connection's row_factory state.
        edge_count = nm.store.conn.execute(
            "SELECT COUNT(*) AS n FROM connections WHERE source_id IN "
            "(SELECT id FROM memories WHERE label LIKE 'chain:%')"
            "   OR target_id IN "
            "(SELECT id FROM memories WHERE label LIKE 'chain:%')"
        ).fetchone()[0]
        expected_edges = 2 * self.n_chains
        print(f"  chain edges    : {edge_count} (expected {expected_edges})")
        assert edge_count == expected_edges, (
            f"chain edge count mismatch: got {edge_count}, expected "
            f"{expected_edges} (2 per chain: A→B and B→C)"
        )
        # No A↔C shortcuts allowed for any chain.
        for a_id, _b_id, c_id in chain_ids:
            row = nm.store.conn.execute(
                "SELECT COUNT(*) AS n FROM connections "
                "WHERE (source_id = ? AND target_id = ?) "
                "   OR (source_id = ? AND target_id = ?)",
                (a_id, c_id, c_id, a_id),
            ).fetchone()
            assert row["n"] == 0, (
                f"unexpected A→C shortcut in chain (a_id={a_id}, c_id={c_id}); "
                "the multihop test would be trivialised"
            )

        results: Dict[str, Any] = {
            "setup": {
                "chains": self.n_chains,
                "memories": len(memories),
                "explicit_edges": edge_count,
            }
        }

        # 1. Raw cosine baseline — should be near zero on hop-2.
        results["raw_cosine"] = _measure_pipeline("raw_cosine", raw_recall, queries, self.k)
        print(f"  raw_cosine     : R@{self.k}={results['raw_cosine']['recall_C_at_k']}  "
              f"MRR={results['raw_cosine']['mrr_C']}")

        # 2. mazemaker semantic mode — also should miss without traversal.
        nm._retrieval_mode = "semantic"
        results["nm_semantic"] = _measure_pipeline(
            "nm_semantic", lambda q, k=10: nm.recall(q, k=k), queries, self.k
        )
        print(f"  nm_semantic    : R@{self.k}={results['nm_semantic']['recall_C_at_k']}  "
              f"MRR={results['nm_semantic']['mrr_C']}")

        # 3. mazemaker skynet — multi-channel + PPR may pull C in.
        nm._retrieval_mode = "skynet"
        results["nm_skynet"] = _measure_pipeline(
            "nm_skynet", lambda q, k=10: nm.recall(q, k=k), queries, self.k
        )
        print(f"  nm_skynet      : R@{self.k}={results['nm_skynet']['recall_C_at_k']}  "
              f"MRR={results['nm_skynet']['mrr_C']}")

        # 4. recall_multihop — explicit graph-traversal recall.
        nm._retrieval_mode = "semantic"
        results["nm_multihop"] = _measure_pipeline(
            "nm_multihop",
            lambda q, k=10: nm.recall_multihop(q, k=k, hops=2),
            queries, self.k,
        )
        print(f"  nm_multihop    : R@{self.k}={results['nm_multihop']['recall_C_at_k']}  "
              f"MRR={results['nm_multihop']['mrr_C']}")

        # 5. think() from A — the actual graph traversal API. Drives think
        # via a small adapter that first finds A by recall, then expands.
        def think_via_a(q: str, k: int = 10) -> List[Dict[str, Any]]:
            seeds = nm.recall(q, k=3)
            if not seeds:
                return []
            # F74 fix (audit 2026-05-13): recall may return either an
            # `id` or `memory_id` field, and the value may be int OR
            # string. Use whichever is present and only int() if it
            # looks numeric.
            seed = seeds[0]
            raw = seed.get("id") if seed.get("id") is not None else seed.get("memory_id")
            if raw is None:
                return []
            try:
                start_id = int(raw)
            except (TypeError, ValueError):
                start_id = raw  # leave as-is; nm.think handles strings on PG
            activated = nm.think(start_id, depth=3, decay=0.85)
            # Render labels into content so anchor-match works.
            ids = [a["id"] for a in activated]
            mems = nm.store.get_many(ids, include_embedding=False) if ids else {}
            out: List[Dict[str, Any]] = []
            for a in activated:
                mid = a["id"]
                m = mems.get(mid, {})
                out.append({
                    "id": mid,
                    "label": a.get("label", m.get("label", "")),
                    "content": m.get("content", ""),
                    "activation": a.get("activation", 0),
                })
            return out[:k]

        results["nm_think"] = _measure_pipeline("nm_think", think_via_a, queries, self.k)
        print(f"  nm_think       : R@{self.k}={results['nm_think']['recall_C_at_k']}  "
              f"MRR={results['nm_think']['mrr_C']}")

        # 6. Negative control: scramble the chain edges.
        # Drop EVERY edge touching any chain memory (covers source_id and
        # target_id, since add_connection canonicalises endpoints), verify
        # the chain is fully disconnected, then re-add EXACTLY the same
        # number of edges (2 * n_chains) between RANDOMLY paired chain
        # memories — so any multihop lift here must be coincidental, not
        # graph-structured. If multihop's gain was graph-driven, this run
        # should collapse to ~raw_cosine.
        try:
            nm.store.conn.execute(
                "DELETE FROM connections WHERE source_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%') "
                "   OR target_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%')"
            )
            nm.store.conn.commit()
            # Hard-verify the chain is actually disconnected before we add
            # the random edges. If any leftover edges (e.g. from a future
            # auto_connect bug) still touch chain memories, fail loudly so
            # the control isn't silently invalidated.
            # F73 fix: index by position, not Row factory key.
            residual = nm.store.conn.execute(
                "SELECT COUNT(*) AS n FROM connections WHERE source_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%') "
                "   OR target_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%')"
            ).fetchone()[0]
            assert residual == 0, (
                f"shuffle control: chain not fully disconnected after delete "
                f"({residual} edges remain) — random pairings would not be "
                "the only graph signal"
            )
            # Invalidate the in-memory graph cache so think()/recall_multihop
            # re-read the (now empty/random) edge set from SQLite.
            # F35 fix (audit 2026-05-13): the previous code silently
            # swallowed cache-invalidation failures, leaving stale chain
            # edges in the in-memory graph and producing false-positive
            # shuffle-control numbers. RAISE on failure so it's loud
            # (the suite cannot honestly proceed without invalidation).
            if hasattr(nm, "_graph_nodes"):
                nm._graph_nodes.clear()
            else:
                raise RuntimeError(
                    "graph_reasoning: nm._graph_nodes not found — the engine's "
                    "in-memory graph cache cannot be invalidated; shuffle "
                    "control would be measuring the cached chain, not the "
                    "randomised edges. Update the suite to match the engine's "
                    "current cache attribute."
                )

            # Random pairings: same edge count as the real chain (2*n).
            rng_ctrl = random.Random(self.seed + 7)
            shuffled = list(memory_ids)
            rng_ctrl.shuffle(shuffled)
            target_edge_count = 2 * self.n_chains
            added: set[Tuple[int, int]] = set()
            attempts = 0
            while len(added) < target_edge_count and attempts < target_edge_count * 20:
                attempts += 1
                a, b = rng_ctrl.sample(shuffled, 2)
                key = (a, b) if a < b else (b, a)
                if key in added:
                    continue
                # Also forbid recreating any of the real chain edges by
                # accident — would partially restore the genuine signal.
                real_pair = False
                for ca, cb, cc in chain_ids:
                    if key == tuple(sorted((ca, cb))) or key == tuple(sorted((cb, cc))):
                        real_pair = True
                        break
                if real_pair:
                    continue
                nm.store.add_connection(key[0], key[1], weight=0.7, edge_type="references")
                added.add(key)

            ctrl_edge_count = nm.store.conn.execute(
                "SELECT COUNT(*) AS n FROM connections WHERE source_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%') "
                "   OR target_id IN "
                "(SELECT id FROM memories WHERE label LIKE 'chain:%')"
            ).fetchone()["n"]
            assert ctrl_edge_count == target_edge_count, (
                f"shuffle control: built {ctrl_edge_count} edges, "
                f"expected {target_edge_count}"
            )
            print(f"  shuffle edges  : {ctrl_edge_count} (random pairings)")

            results["nm_multihop_shuffled"] = _measure_pipeline(
                "nm_multihop_shuffled",
                lambda q, k=10: nm.recall_multihop(q, k=k, hops=2),
                queries, self.k,
            )
            print(f"  multihop[ctrl] : R@{self.k}={results['nm_multihop_shuffled']['recall_C_at_k']}  "
                  f"MRR={results['nm_multihop_shuffled']['mrr_C']}")
        except Exception as e:
            results["nm_multihop_shuffled"] = {"error": str(e)}

        # Summary deltas — the "graph lift"
        raw_r = results["raw_cosine"]["recall_C_at_k"]
        results["analysis"] = {
            "graph_lift_vs_raw": {
                "nm_semantic":  round(results["nm_semantic"]["recall_C_at_k"] - raw_r, 4),
                "nm_skynet":    round(results["nm_skynet"]["recall_C_at_k"]   - raw_r, 4),
                "nm_multihop":  round(results["nm_multihop"]["recall_C_at_k"] - raw_r, 4),
                "nm_think":     round(results["nm_think"]["recall_C_at_k"]    - raw_r, 4),
            },
            "shuffle_collapse": (
                round(
                    results["nm_multihop"]["recall_C_at_k"]
                    - results.get("nm_multihop_shuffled", {}).get("recall_C_at_k", raw_r),
                    4,
                )
                if isinstance(results.get("nm_multihop_shuffled"), dict)
                and "recall_C_at_k" in results["nm_multihop_shuffled"]
                else None
            ),
            "interpretation": (
                "graph_lift_vs_raw should be > 0 for multihop/think if the "
                "graph is doing real work. shuffle_collapse should be > 0 "
                "(multihop drops back when chain edges are scrambled) — "
                "that proves the lift was edge-driven not semantic-driven."
            ),
        }

        out = self.output_dir / "graph_reasoning_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
