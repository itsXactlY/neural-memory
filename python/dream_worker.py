#!/usr/bin/env python3
"""
Dream Worker — Standalone full-stack dream engine.

Supports SQLite (default) and MSSQL backends. Auto-detects available backend.

Usage:
    python dream_worker.py                     # one-shot dream cycle (auto-detect)
    python dream_worker.py --backend sqlite    # force SQLite
    python dream_worker.py --backend mssql     # force MSSQL
    python dream_worker.py --daemon            # background loop (idle-based)
    python dream_worker.py --phase nrem        # single phase only
    python dream_worker.py --db /path/to.db    # custom SQLite path

Config: reads from config.yaml or env vars.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure plugin dir is on path
_PLUGIN_DIR = Path(__file__).parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

logger = logging.getLogger("dream_worker")


# ---------------------------------------------------------------------------
# Embedding Provider (sentence-transformers only — full stack)
# ---------------------------------------------------------------------------

class EmbedProvider:
    """Sentence-transformers embedding provider for dream worker.

    Uses the same cache as embed_provider.py: ~/.neural_memory/models/
    """

    MODEL_NAME = "BAAI/bge-m3"
    MODEL_DIR = Path.home() / ".neural_memory" / "models"

    _shared_model = None

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        # Reuse shared model if already loaded
        if EmbedProvider._shared_model is not None:
            self._model = EmbedProvider._shared_model
            return

        from sentence_transformers import SentenceTransformer

        safe_name = self.MODEL_NAME.replace("/", "--")
        cached = self.MODEL_DIR / f"models--{safe_name}"
        is_cached = cached.exists() and (cached / "config.json").exists()

        if is_cached:
            # Find snapshot path
            snapshot_path = None
            refs_main = cached / "refs" / "main"
            if refs_main.exists():
                snapshot_hash = refs_main.read_text().strip()
                snap = cached / "snapshots" / snapshot_hash
                if (snap / "config.json").exists():
                    snapshot_path = str(snap)
            if snapshot_path is None:
                snapshots_dir = cached / "snapshots"
                if snapshots_dir.exists():
                    for snap in snapshots_dir.iterdir():
                        if (snap / "config.json").exists():
                            snapshot_path = str(snap)
                            break
            if snapshot_path:
                logger.info("Loading %s from local cache (%s)...", self.MODEL_NAME, snapshot_path)
                self._model = SentenceTransformer(snapshot_path)
            else:
                logger.warning("Cache dir exists but no snapshot found, downloading...")
                self._model = SentenceTransformer(
                    self.MODEL_NAME,
                    cache_folder=str(self.MODEL_DIR),
                )
        else:
            logger.info("Downloading %s (first time, ~2.2GB)...", self.MODEL_NAME)
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                cache_folder=str(self.MODEL_DIR),
            )
        EmbedProvider._shared_model = self._model
        logger.info("Embedding model ready: %s (%sd)", self.MODEL_NAME, self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> List[float]:
        self._load()
        vec = self._model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vecs = self._model.encode(texts, show_progress_bar=False,
                                   batch_size=64)
        return [v.tolist() for v in vecs]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Dream Worker
# ---------------------------------------------------------------------------

class DreamWorker:
    """Full-stack dream engine — SQLite first, MSSQL cold storage."""

    def __init__(self, backend: str = "auto", db_path: str = "", mssql_config: Optional[dict] = None):
        if backend == "auto":
            backend = self._detect_backend()

        if backend == "sqlite":
            from dream_engine import SQLiteDreamBackend
            db_path = db_path or os.path.expanduser("~/.neural_memory/memory.db")
            self.store = SQLiteDreamBackend(db_path)
            self._backend_type = "sqlite"
            logger.info("DreamWorker: SQLite backend (%s)", db_path)
        else:
            from dream_mssql_store import DreamMSSQLStore
            if mssql_config:
                self.store = DreamMSSQLStore.from_config(mssql_config)
            else:
                self.store = DreamMSSQLStore()
            self._backend_type = "mssql"
            logger.info("DreamWorker: MSSQL backend")

        from embed_provider import EmbeddingProvider as _EP
        self.embedder = _EP()
        self._embedding_cache: OrderedDict[int, List[float]] = OrderedDict()
        # Prevent unbounded RAM growth in daemon mode.
        self._embedding_cache_max = max(128, int(os.environ.get("DREAM_EMBED_CACHE_MAX", "2048")))

    @staticmethod
    def _detect_backend() -> str:
        """Auto-detect: try MSSQL, fallback to SQLite."""
        try:
            import pyodbc
            from dream_mssql_store import DreamMSSQLStore
            store = DreamMSSQLStore()
            store.close()
            return "mssql"
        except Exception:
            pass
        return "sqlite"

    def close(self):
        self.store.close()

    # -- Embedding helpers ---------------------------------------------------

    def _get_embedding(self, memory_id: int, content: str) -> Optional[List[float]]:
        """Get embedding for a memory, with bounded caching."""
        if memory_id in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(memory_id)
            return self._embedding_cache[memory_id]
        if not content or not content.strip():
            return None
        try:
            emb = self.embedder.embed(content[:512])
            # Bounded cache (FIFO by insertion order) — evict BEFORE adding.
            while len(self._embedding_cache) >= self._embedding_cache_max:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[memory_id] = emb
            return emb
        except Exception as e:
            logger.debug("Embed failed for memory %d: %s", memory_id, e)
            return None

    def _similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity with strict dim-equality.

        Mixed-dim embeddings (different models in the same DB) silently
        zip-truncated to the shorter vector and produced a partial dot
        product — pure noise that biased dream-engine bridge discovery
        toward whichever model happened to win the truncation. Treat
        mismatch as zero signal so downstream phases skip the pair.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0

    # -- Phase 1: NREM -------------------------------------------------------

    def phase_nrem(self, limit: int = 100) -> Dict[str, Any]:
        """NREM: Replay recent memories, strengthen active connections.

        1. Load recent memories from MSSQL
        2. Compute embeddings batch
        3. For each memory, find connections where the OTHER end is similar
        4. Strengthen those connections (active replay reinforces them)
        5. Weak connections (< 0.1) get pruned
        """
        logger.info("NREM: loading %d recent memories...", limit)
        memories = self.store.get_recent_memories(limit)
        if not memories:
            return {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}

        session_id = self.store.start_session("nrem")
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}

        try:
            # Build embedding index
            logger.info("NREM: computing %d embeddings...", len(memories))
            contents = [m["content"][:512] for m in memories if m["content"]]
            mem_ids = [m["id"] for m in memories if m["content"]]

            embeddings = self.embedder.embed_batch(contents)
            embed_map = dict(zip(mem_ids, embeddings))

            # Batch-fetch every connection touching any memory in mem_ids in
            # one SQL round-trip instead of one query per memory. The previous
            # loop fired 100 SELECTs per NREM phase; on a large connections
            # table that latency dominated.
            logger.info("NREM: finding activated connections...")
            activated_edges = set()
            cursor = self.store.conn.cursor()
            mid_set = {m for m in mem_ids if m in embed_map}
            if mid_set:
                placeholders = ",".join("?" * len(mid_set))
                ids_list = list(mid_set)
                cursor.execute(
                    f"SELECT source_id, target_id, weight FROM connections "
                    f"WHERE source_id IN ({placeholders}) "
                    f"   OR target_id IN ({placeholders})",
                    ids_list + ids_list,
                )
                for src, tgt, _w in cursor.fetchall():
                    # An edge counts as activated if either endpoint's
                    # embedding is in our hot set AND the other endpoint's
                    # embedding is similar above 0.4. We test BOTH orientations
                    # so we don't miss when only one side is in mid_set.
                    activated = False
                    if src in mid_set and tgt in embed_map:
                        if self._similarity(embed_map[src], embed_map[tgt]) > 0.4:
                            activated = True
                    if not activated and tgt in mid_set and src in embed_map:
                        if self._similarity(embed_map[tgt], embed_map[src]) > 0.4:
                            activated = True
                    if activated:
                        activated_edges.add((min(src, tgt), max(src, tgt)))
            stats["processed"] = len(mid_set)

            # Strengthen activated connections — single executemany instead
            # of N individually-prepared UPDATEs.
            logger.info("NREM: strengthening %d connections...", len(activated_edges))
            if activated_edges:
                cursor.executemany(
                    "UPDATE connections SET weight = CASE "
                    "WHEN weight + 0.05 > 1.0 THEN 1.0 ELSE weight + 0.05 END "
                    "WHERE source_id = ? AND target_id = ?",
                    list(activated_edges),
                )
            self.store.conn.commit()
            stats["strengthened"] = len(activated_edges)

            # Prune
            stats["pruned"] = self.store.prune_weak(0.05)

            logger.info("NREM: %d processed, %d strengthened, %d weakened, %d pruned",
                        stats["processed"], stats["strengthened"],
                        stats["weakened"], stats["pruned"])
        except Exception as e:
            logger.debug("NREM phase error: %s", e)
        finally:
            try:
                self.store.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("NREM finish_session failed: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def phase_rem(self, limit: int = 100) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Compute embeddings for all isolated memories
        3. Compare against ALL memories to find unconnected but similar
        4. Create bridge connections
        """
        logger.info("REM: finding isolated memories...")
        isolated = self.store.get_isolated_memories(max_connections=3, limit=limit)
        if not isolated:
            return {"explored": 0, "bridges": 0, "rejected": 0}

        session_id = self.store.start_session("rem")
        stats = {"explored": 0, "bridges": 0, "rejected": 0}

        try:
            # Embed isolated memories
            logger.info("REM: embedding %d isolated memories...", len(isolated))
            iso_embeds: Dict[int, List[float]] = {}
            for mem in isolated:
                emb = self._get_embedding(mem["id"], mem["content"])
                if emb:
                    iso_embeds[mem["id"]] = emb

            # Get a sample of ALL memories for comparison
            all_recent = self.store.get_recent_memories(500)
            logger.info("REM: embedding %d comparison memories...", len(all_recent))
            comp_embeds: Dict[int, List[float]] = {}
            for mem in all_recent:
                if mem["id"] in iso_embeds:
                    continue
                emb = self._get_embedding(mem["id"], mem["content"])
                if emb:
                    comp_embeds[mem["id"]] = emb

            # Vectorised bridge discovery: a single (|iso| x |comp|) matmul
            # replaces |iso| * |comp| Python sum-zip cosines. With the
            # default 50 isolated × 500 comparison memories at 1024-d, that
            # was ~25K Python loops per REM phase; the numpy path runs in
            # one matrix op. Falls back to the pure-Python loop if numpy
            # isn't available, which preserves the original behaviour.
            try:
                import numpy as np

                if not iso_embeds or not comp_embeds:
                    raise RuntimeError("nothing to compare")
                # Filter dim-mismatched embeddings (cf. iter 11's cosine
                # guard) so the stack/matmul doesn't blow up.
                iso_dim = len(next(iter(iso_embeds.values())))
                iso_items = [(mid, e) for mid, e in iso_embeds.items() if len(e) == iso_dim]
                comp_items = [(mid, e) for mid, e in comp_embeds.items() if len(e) == iso_dim]
                if not iso_items or not comp_items:
                    raise RuntimeError("dim mismatch only")

                iso_ids_arr = [mid for mid, _ in iso_items]
                comp_ids_arr = [mid for mid, _ in comp_items]
                iso_mat = np.asarray([e for _, e in iso_items], dtype=np.float32)
                comp_mat = np.asarray([e for _, e in comp_items], dtype=np.float32)
                # Row-normalise both sides so the matmul yields cosine
                # similarity directly.
                iso_norms = np.linalg.norm(iso_mat, axis=1, keepdims=True).clip(min=1e-12)
                comp_norms = np.linalg.norm(comp_mat, axis=1, keepdims=True).clip(min=1e-12)
                iso_n = iso_mat / iso_norms
                comp_n = comp_mat / comp_norms
                sims = iso_n @ comp_n.T  # (|iso|, |comp|)

                for i, iso_id in enumerate(iso_ids_arr):
                    row = sims[i]
                    # Mask the [0.3, 0.95] band, take top-3 by similarity.
                    mask = (row > 0.3) & (row < 0.95)
                    idxs = np.where(mask)[0]
                    if idxs.size == 0:
                        stats["explored"] += 1
                        continue
                    # Sort the surviving candidates descending by similarity.
                    top = idxs[np.argsort(-row[idxs])[:3]]
                    for j in top:
                        comp_id = comp_ids_arr[int(j)]
                        sim = float(row[int(j)])
                        bridge_weight = round(sim * 0.3, 3)
                        self.store.add_bridge(iso_id, comp_id, bridge_weight)
                        self.store.log_connection_change(
                            iso_id, comp_id, 0.0, bridge_weight, "rem_bridge"
                        )
                        stats["bridges"] += 1
                    stats["explored"] += 1
            except Exception:
                # Pure-Python fallback (numpy missing, dim mismatch only,
                # or any other failure). Original behaviour preserved.
                for iso_id, iso_emb in iso_embeds.items():
                    similarities = []
                    for comp_id, comp_emb in comp_embeds.items():
                        sim = self._similarity(iso_emb, comp_emb)
                        if 0.3 < sim < 0.95:
                            similarities.append((comp_id, sim))
                    similarities.sort(key=lambda x: -x[1])
                    for comp_id, sim in similarities[:3]:
                        bridge_weight = round(sim * 0.3, 3)
                        self.store.add_bridge(iso_id, comp_id, bridge_weight)
                        self.store.log_connection_change(
                            iso_id, comp_id, 0.0, bridge_weight, "rem_bridge"
                        )
                        stats["bridges"] += 1
                    stats["explored"] += 1

            logger.info("REM: %d explored, %d bridges created",
                        stats["explored"], stats["bridges"])
        except Exception as e:
            logger.debug("REM phase error: %s", e)
        finally:
            try:
                self.store.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("REM finish_session failed: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction."""
        logger.info("Insight: building graph from %s connections...", self._backend_type)
        stats = {"communities": 0, "bridges": 0, "insights": 0}
        session_id = self.store.start_session("insight")

        try:
            edges = self.store.get_connections()
            if not edges:
                return stats

            # Build adjacency
            adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            nodes = set()
            for e in edges:
                s, t, w = e["source_id"], e["target_id"], e["weight"]
                if w >= 0.3:
                    adj[s].append((t, w))
                    adj[t].append((s, w))
                    nodes.add(s)
                    nodes.add(t)

            # Connected components (BFS)
            visited = set()
            communities: List[List[int]] = []
            for node in nodes:
                if node in visited:
                    continue
                component = []
                queue = [node]
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.append(curr)
                    for neighbor, _ in adj.get(curr, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
                communities.append(component)

            stats["communities"] = len(communities)
            logger.info("Insight: found %d communities", len(communities))

            # Map nodes to communities
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # Find bridge nodes
            bridge_nodes = set()
            for e in edges:
                if e["weight"] < 0.3:
                    continue
                s_comm = node_to_comm.get(e["source_id"], -1)
                t_comm = node_to_comm.get(e["target_id"], -1)
                if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                    bridge_nodes.add(e["source_id"])
                    bridge_nodes.add(e["target_id"])

            stats["bridges"] = len(bridge_nodes)

            # Cluster insights (only for communities >= 5 members)
            for i, comm in enumerate(communities):
                if len(comm) < 5:
                    continue
                theme = self._extract_theme(comm)
                confidence = min(len(comm) / 20.0, 1.0)
                content = f"Cluster of {len(comm)} memories: {theme}"
                self.store.add_insight(session_id, "cluster", comm[0], content, confidence)
                stats["insights"] += 1

            # Bridge insights — twin of the iter 18 fix on the in-process
            # dream engine. Use the adjacency map already built above
            # instead of rescanning the full edge list per bridge node;
            # adj[node] yields the node's neighbours directly so the loop
            # collapses from O(|bridges| * |edges|) to O(B + sum-of-degrees).
            for bnode in bridge_nodes:
                bridging_communities = set()
                for neighbour, _w in adj.get(bnode, ()):
                    bridging_communities.add(node_to_comm.get(neighbour, -1))
                bridging_communities.discard(-1)

                if len(bridging_communities) >= 3:
                    content = f"Bridge connecting {len(bridging_communities)} communities, memory #{bnode}"
                    self.store.add_insight(session_id, "bridge", bnode, content, 0.8)
                    stats["insights"] += 1

            logger.info("Insight: %d communities, %d bridges, %d insights",
                        stats["communities"], stats["bridges"], stats["insights"])
        except Exception as e:
            logger.debug("Insight phase error: %s", e)
        finally:
            try:
                self.store.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("Insight finish_session failed: %s", e)

        return stats

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs via keyword frequency."""
        placeholders = ",".join(str(n) for n in node_ids[:100])
        try:
            cursor = self.store.conn.cursor()
            if self._backend_type == "mssql":
                cursor.execute(
                    f"SELECT TOP 50 content FROM memories WHERE id IN ({placeholders})"
                )
            else:
                cursor.execute(
                    f"SELECT content FROM memories WHERE id IN ({placeholders}) LIMIT 50"
                )
            contents = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception:
            return f"{len(node_ids)} memories"

        word_freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "it", "its", "this", "that", "user", "assistant", "and", "or",
            "but", "not", "if", "then", "so", "just", "also", "very",
            "really", "like", "get", "got", "want", "need", "think",
            "know", "see", "look", "make", "let", "use", "still",
        }
        for c in contents:
            for w in c.lower().split():
                w = w.strip(".,!?;:'\"()[]{}#@")
                if len(w) > 3 and w not in stopwords:
                    word_freq[w] += 1

        top = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
        return ", ".join(w for w, _ in top) if top else "mixed topics"

    # -- Full cycle ----------------------------------------------------------

    def dream(self, phase: str = "all") -> Dict[str, Any]:
        """Run a dream cycle (or specific phase)."""
        start = time.time()
        result: Dict[str, Any] = {}

        try:
            if phase in ("all", "nrem"):
                result["nrem"] = self.phase_nrem()

            if phase in ("all", "rem"):
                result["rem"] = self.phase_rem()

            if phase in ("all", "insight"):
                result["insights"] = self.phase_insights()

            result["duration"] = time.time() - start
            return result
        finally:
            # Hard cleanup to prevent cache accumulation in long-lived daemon mode.
            cached = len(self._embedding_cache)
            self._embedding_cache.clear()
            if cached:
                logger.debug("Dream cycle cache cleanup: cleared %d embeddings", cached)
            try:
                import gc
                gc.collect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dream Worker — SQLite + MSSQL dual-backend")
    parser.add_argument("--phase", default="all", choices=["all", "nrem", "rem", "insight"])
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--idle", type=int, default=300, help="Idle threshold in seconds")
    parser.add_argument("--limit", type=int, default=200, help="Max memories per phase")
    parser.add_argument("--backend", default="auto", choices=["auto", "sqlite", "mssql"],
                        help="Backend: sqlite (default), mssql, or auto-detect")
    parser.add_argument("--db", default="", help="SQLite database path (default: ~/.neural_memory/memory.db)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Keep model on GPU during dream cycles — prevent idle eject timer from
    # evicting during long non-embedding phases (e.g. 57s Insight BFS).
    os.environ["EMBED_IDLE_TIMEOUT"] = "0"

    worker = DreamWorker(backend=args.backend, db_path=args.db)

    try:
        if args.daemon:
            logger.info("Dream daemon started (idle=%ds)", args.idle)
            last_activity = time.time()
            while True:
                time.sleep(30)
                idle = time.time() - last_activity
                if idle >= args.idle:
                    logger.info("Dream triggered (idle=%.0fs)", idle)
                    result = worker.dream()
                    logger.info("Dream complete: %.1fs", result["duration"])
                    last_activity = time.time()
        else:
            result = worker.dream(args.phase)
            print(f"\nDream complete in {result['duration']:.1f}s:")
            for phase, stats in result.items():
                if phase == "duration":
                    continue
                print(f"  {phase}: {stats}")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
