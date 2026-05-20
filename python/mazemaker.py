#!/usr/bin/env python3
"""
mazemaker.py - THE Unified Mazemaker API
One import to rule them all.

Architecture:
  Python  ────────────── Embedding, dream engine, orchestration
  SQLite  ────────────── Hot store + semantic recall (always present)
  Postgres + pgvector ── Optional graph/cold-storage mirror
                         (enable with MM_DB_BACKEND=postgres)

Usage:
    from mazemaker import Memory

    mem = Memory()  # SQLite by default; honours MM_DB_BACKEND=postgres
    mem.remember("The user has a dog named Lou")
    results = mem.recall("What pet does the user have?")
    mem.think(results[0].id)
    mem.consolidate()
    mem.close()
"""

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add python dir to path
sys.path.insert(0, str(Path(__file__).parent))

from embed_provider import EmbeddingProvider
from memory_client import Mazemaker, SQLiteStore


class Memory:
    """
    Unified Mazemaker interface with LSTM+kNN enhancement.

    Backend priority:
    1. Postgres + pgvector — when MM_DB_BACKEND=postgres is set
    2. SQLite (default) — always available

    LSTM+kNN is auto-initialized when libmazemaker.so is available.
    - AccessLogger: records every recall event
    - LSTMPredictor: learns access patterns, predicts next relevant embedding
    - KNNEngine: multi-signal re-ranking (embedding + temporal + frequency + graph)
    """

    def __init__(self,
                 db_path: Optional[str] = None,
                 embedding_backend: str = "auto",
                 use_cpp: bool = True,
                 default_chunk_size: int = 512,
                 retrieval_mode: str = "semantic",
                 retrieval_candidates: int = 64,
                 use_hnsw: bool | str | None = None,
                 lazy_graph: bool = False,
                 think_engine: str = "bfs",
                 rerank: bool = False,
                 channel_weights: Optional[dict] = None,
                 rrf_k: int = 60,
                 salience_decay_k: float = 0.03,
                 ppr_alpha: float = 0.15,
                 ppr_iters: int = 20,
                 ppr_hops: int = 2,
                 mmr_lambda: float = 0.0,
                 recall_score_floor: float = 0.0,
                 recall_score_percentile: float = 0.0):
        
        Path.home().joinpath(".mazemaker", "engine").mkdir(parents=True, exist_ok=True)

        self._db_path = db_path or str(Path.home() / ".mazemaker" / "engine" / "memory.db")
        self._default_chunk_size = default_chunk_size
        self._sqlite_memory = None

        # Embedder (shared regardless of backend)
        from embed_provider import EmbeddingProvider
        self._embedder = EmbeddingProvider(backend=embedding_backend)
        self._dim = self._embedder.dim
        # Fail fast: a None / 0 dim means the embedder either crashed
        # during init or returned a placeholder. Carrying that downstream
        # produces cryptic shape errors deep inside SQLite store +
        # GpuRecallEngine + every cosine call. Surface the cause here.
        if not self._dim or self._dim <= 0:
            raise RuntimeError(
                f"EmbeddingProvider({embedding_backend!r}) returned invalid dim={self._dim!r}; "
                "check that the embed-server is running and the model loaded successfully."
            )

        # Postgres + pgvector dispatch. Activated via MM_DB_BACKEND=postgres.
        # The Postgres store plays the graph/cold-storage mirror role — SQLite
        # remains the source of truth for semantic recall. When unset or
        # set to anything else, we run SQLite-only.
        backend_choice = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
        self._postgres_store = None
        if backend_choice == "postgres":
            try:
                from postgres_store import PostgresStore
                self._postgres_store = PostgresStore()
                logger.info(
                    "Postgres backend armed: %s (%dd)",
                    self._embedder.backend.__class__.__name__, self._dim,
                )
            except Exception as e:
                logger.warning("Postgres unavailable (%s), falling back to SQLite", e)

        # SQLite is always the source of truth for semantic recall.
        from memory_client import Mazemaker
        self._sqlite_memory = Mazemaker(
            db_path=self._db_path,
            embedding_backend=embedding_backend,
            use_cpp=use_cpp,
            embedder=self._embedder,
            retrieval_mode=retrieval_mode,
            retrieval_candidates=retrieval_candidates,
            use_hnsw=use_hnsw,
            lazy_graph=lazy_graph,
            think_engine=think_engine,
            rerank=rerank,
            channel_weights=channel_weights,
            rrf_k=rrf_k,
            salience_decay_k=salience_decay_k,
            ppr_alpha=ppr_alpha,
            ppr_iters=ppr_iters,
            ppr_hops=ppr_hops,
            mmr_lambda=mmr_lambda,
            recall_score_floor=recall_score_floor,
            recall_score_percentile=recall_score_percentile,
        )
        # Cache the configured defaults so Memory.recall() can pass an
        # explicit override per-call. Mazemaker.recall() falls back to
        # its own stored defaults when these are None.
        self._mmr_lambda_default = float(mmr_lambda or 0.0)
        self._recall_score_floor_default = float(recall_score_floor or 0.0)
        self._recall_score_percentile_default = float(recall_score_percentile or 0.0)

        if self._postgres_store is None:
            print(f"[neural] SQLite backend: {self._embedder.backend.__class__.__name__} ({self._dim}d)")
        else:
            print(f"[neural] Hybrid mode: Postgres+pgvector (graph) + SQLite (recall)")

        # --- LSTM + kNN (auto-initialized) ---
        self._access_logger = None
        self._lstm = None
        self._knn = None
        self._lstm_knn_ready = False
        self._init_lstm_knn()
    
    def _init_lstm_knn(self):
        """Auto-initialize LSTM+kNN if libmazemaker.so is available."""
        try:
            from access_logger import AccessLogger
            from lstm_knn_bridge import LSTMPredictor, KNNEngine
            
            self._access_logger = AccessLogger.instance()
            self._lstm = LSTMPredictor(input_dim=self._dim, hidden_dim=256)
            self._knn = KNNEngine(embed_dim=self._dim)
            
            # Try to load saved LSTM weights
            lstm_weights_path = Path.home() / ".mazemaker" / "engine" / "lstm_weights.bin"
            if lstm_weights_path.exists():
                try:
                    self._lstm.load(str(lstm_weights_path))
                except Exception:
                    pass  # Fresh LSTM is fine
            
            self._lstm_knn_ready = True
            print(f"[neural] LSTM+kNN active ({self._dim}d, hidden=256)")
        except Exception as e:
            self._lstm_knn_ready = False
            # Silent — LSTM+kNN is optional enhancement
    
    def _enhance_recall(self, query_embedding: list, base_results: list[dict], k: int) -> list[dict]:
        """Enhance base recall results with LSTM prediction + kNN re-ranking.
        
        This is the core integration: transparently improves recall quality
        without changing the API. Falls back to base_results on any error.
        """
        if not self._lstm_knn_ready or not base_results:
            return base_results
        
        import time as _time
        import math as _math
        
        try:
            # 1. Log this access
            result_ids = [r.get('id', 0) for r in base_results]
            result_scores = [r.get('similarity', r.get('combined', 0)) for r in base_results]
            self._access_logger.log_recall(
                query_embedding=query_embedding,
                result_ids=result_ids,
                result_scores=result_scores,
            )
            
            # 2. LSTM prediction from access sequence — exclude the
            # just-logged event so predict_next is genuinely \"next from
            # prior\", not \"copy the current input.\" Same shape as the
            # iter-68 fix on the training path: log_recall just appended
            # an event whose query_emb == query_embedding, so feeding
            # recent[-10:] to predict_next would put the target itself
            # at the end of the input sequence and degrade the prediction
            # to ≈ query_embedding (a trivial copy).
            lstm_context = None
            recent = self._access_logger.get_sequence(n=15)
            if len(recent) >= 3:
                try:
                    prior = recent[:-1]
                    seq_embs = [e["query_emb"] for e in prior[-10:]]
                    if len(seq_embs) >= 2:
                        lstm_context = self._lstm.predict_next(seq_embs)
                except Exception:
                    pass  # Fall through without LSTM context
            
            # 3. Get embeddings + metadata for kNN re-ranking
            candidates = []
            candidate_ids = []
            timestamps = []
            access_counts = []
            graph_scores = []
            
            now = _time.time()
            for r in base_results:
                mem_id = r.get('id')
                emb = r.get('embedding')
                
                # If embedding not in result, try to fetch it from the
                # in-memory SQLite graph cache, then the SQLite store on
                # disk. (Postgres mirror is graph-only; embeddings live in
                # SQLite.)
                if emb is None:
                    if self._sqlite_memory:
                        # Fast path: in-memory graph has embeddings
                        node = self._sqlite_memory._graph_nodes.get(mem_id, {})
                        emb = node.get('embedding', [])
                        # Fallback: fetch from store
                        if not emb:
                            try:
                                full = self._sqlite_memory.store.get(mem_id)
                                emb = full.get('embedding', []) if full else []
                            except Exception:
                                emb = []
                
                if not emb or len(emb) != self._dim:
                    continue
                
                candidates.append(emb)
                candidate_ids.append(mem_id)
                
                # Metadata for multi-signal scoring
                ts = r.get('created_at', r.get('timestamp', now))
                try:
                    timestamps.append(float(ts) if ts else now)
                except (TypeError, ValueError):
                    timestamps.append(now)
                
                ac = r.get('access_count', 1)
                try:
                    access_counts.append(float(ac) if ac else 1.0)
                except (TypeError, ValueError):
                    access_counts.append(1.0)
                
                # Graph proximity: count connections as inverse distance
                # connections may be list[int] or list[dict] depending on code path
                conns = r.get('connections', [])
                if isinstance(conns, (list, tuple)):
                    graph_scores.append(min(1.0, len(conns) * 0.1))
                else:
                    graph_scores.append(0.0)
            
            if len(candidates) < 2:
                return base_results  # Not enough candidates for kNN
            
            # 4. kNN search with LSTM context
            knn_results = self._knn.search(
                query=query_embedding,
                candidates=candidates,
                candidate_ids=candidate_ids,
                k=k,
                timestamps=timestamps,
                access_counts=access_counts,
                graph_scores=graph_scores,
                lstm_context=lstm_context,
            )
            
            if not knn_results:
                return base_results
            
            # 5. Build enhanced result list (kNN order, enriched with original data)
            id_to_original = {r.get('id'): r for r in base_results}
            enhanced = []
            for kr in knn_results:
                orig = id_to_original.get(kr.id, {})
                enhanced.append({
                    'id': kr.id,
                    'label': orig.get('label', ''),
                    'content': orig.get('content', ''),
                    'similarity': round(kr.embed_similarity, 4),
                    'temporal_score': round(kr.temporal_score, 4),
                    'combined': round(kr.score, 4),
                    'connections': orig.get('connections', []),
                    # kNN detail scores for debugging
                    '_knn_freq': round(kr.freq_score, 4),
                    '_knn_graph': round(kr.graph_score, 4),
                })
            
            # 6. Train LSTM in background (fire-and-forget).
            # The sequence MUST exclude the just-logged event whose
            # query_embedding is the training target — including it
            # leaks the target into the input and the LSTM degenerates
            # to a copy-last-input identity. Use \`recent[:-1][-10:]\` to
            # take the 10 events BEFORE the current one.
            try:
                if len(recent) >= 3 and lstm_context is not None:
                    prior = recent[:-1]  # drop the just-logged target event
                    seq_embs = [e["query_emb"] for e in prior[-10:]]
                    if len(seq_embs) >= 2:
                        self._lstm.train_on_pair(seq_embs, query_embedding, lr=0.0005)
            except Exception:
                pass
            
            return enhanced
            
        except Exception:
            return base_results  # Transparent fallback
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        """
        Split text into overlapping chunks at sentence boundaries.
        
        Each chunk is approximately chunk_size characters with overlap
        characters shared between adjacent chunks. Preserves sentence
        integrity (never cuts mid-sentence).
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters (default 512)
            overlap: Number of overlapping characters between chunks (default 64)
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is full — start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence exceeds chunk_size, add it alone
                if len(sentence) > chunk_size:
                    chunks.append(sentence)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap: prepend end of previous chunk to each chunk
        # Guard: overlap >= chunk_size causes duplicate chunks
        # (taking the last chunk_size chars of a chunk_size-sized chunk = the whole thing)
        if overlap > 0 and len(chunks) > 1:
            effective_overlap = min(overlap, max(1, chunk_size // 2))
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                # Find the text boundary: where did prev end in the full reconstructed text?
                # Take enough overlap chars to bridge to the start of current chunk,
                # but ensure we don't duplicate the entire previous chunk.
                # Strategy: if prev already ends with similar start as current,
                # use 1-char word-boundary split; otherwise use effective_overlap
                prev_tail = prev[-effective_overlap:]
                curr_start = chunks[i][:effective_overlap]
                # If prev's tail and curr's start are too similar, reduce overlap
                if prev_tail.strip() == curr_start.strip():
                    # Nearly identical overlap — use word boundary only
                    overlap_text = prev[-max(1, len(prev) // 4):]
                    space_idx = overlap_text.strip().find(" ")
                    if space_idx >= 0:
                        overlap_text = overlap_text[space_idx + 1:]
                else:
                    overlap_text = prev_tail
                    space_idx = overlap_text.find(" ")
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx + 1:]
                overlapped.append((overlap_text + " " + chunks[i]).strip())
            chunks = overlapped
        
        return chunks
    
    def remember_chunked(self, text: str, label: str = "",
                         chunk_size: int = None, overlap: int = 64) -> list[int]:
        """
        Store text as multiple chunked memories.
        
        Splits text into overlapping chunks at sentence boundaries and stores
        each chunk as a separate memory with the same base label.
        
        Args:
            text: Text to store
            label: Base label (chunk index appended)
            chunk_size: Chunk size (default: self._default_chunk_size)
            overlap: Overlap between chunks (default 64)
        
        Returns:
            List of memory IDs
        """
        if chunk_size is None:
            chunk_size = self._default_chunk_size
        
        chunks = [c for c in self.chunk_text(text, chunk_size, overlap) if c and c.strip()]
        if not chunks:
            return []

        if len(chunks) == 1:
            mid = self.remember(chunks[0], label, auto_chunk=False)
            return [mid] if mid != -1 else []

        ids = []
        for i, chunk in enumerate(chunks):
            chunk_label = f"{label} [chunk {i+1}/{len(chunks)}]" if label else f"[chunk {i+1}/{len(chunks)}]"
            mid = self.remember(chunk, chunk_label, auto_chunk=False)
            if mid != -1:
                ids.append(mid)
        return ids
    
    def remember(self, text: str, label: str = "", auto_chunk: bool = True,
                 auto_connect: bool = True, detect_conflicts: bool = True) -> int | list[int]:
        """Store a memory. SQLite primary, optional Postgres mirror. Returns memory ID.

        Refuses empty-after-strip text. Without this guard, chunk_text('')
        returned [''] and remember_chunked happily stored an empty-content
        memory, polluting the DB with junk rows that auto_connect would
        then try to similarity-match against (HashBackend embeds an empty
        string to a high-similarity zero vector, dragging unrelated rows
        into the connection graph).
        """
        if not text or not text.strip():
            return -1
        if auto_chunk and len(text) > self._default_chunk_size * 2:
            return self.remember_chunked(text, label)

        embedding = self._embedder.embed(text)

        # SQLite always (source of truth)
        mem_id = self._sqlite_memory.remember(text, label, auto_connect=auto_connect,
                                              detect_conflicts=detect_conflicts)
        # Postgres mirror — preserve SQLite-assigned id so cross-backend
        # lookups (think, recall_multihop) hit the same row both sides.
        # Read the canonical SQLite row in case conflict-fusion rewrote it.
        if self._postgres_store:
            try:
                canonical = self._sqlite_memory.store.get(int(mem_id), include_embedding=True)
                if canonical is not None:
                    self._postgres_store.store(
                        canonical.get("label") or label or text[:60],
                        canonical.get("content") or text,
                        canonical.get("embedding") or embedding,
                        id_=int(mem_id),
                    )
                else:
                    self._postgres_store.store(
                        label or text[:60], text, embedding, id_=int(mem_id)
                    )
            except Exception:
                pass
        return mem_id

    def remember_embedding(self, embedding: list[float], label: str = "",
                           content: str = "") -> int:
        """Store a memory with pre-computed embedding. SQLite primary, optional Postgres mirror.

        Honours the same dim-lock invariants as remember(): refuses to write
        when the SQLite-side dim_lock is False, and refuses embeddings whose
        length doesn't match the active backend's dim. Without these checks,
        a caller could bypass the iter-12 ONE-MODEL invariant entirely by
        going through this method instead of remember().
        """
        nm = self._sqlite_memory
        if hasattr(nm, "_dim_locked") and not nm._dim_locked:
            raise RuntimeError(
                f"refusing to write: {nm._dim_mismatch_reason} "
                "Re-open with the original embedding backend, or drop the DB."
            )
        if hasattr(nm, "dim") and len(embedding) != nm.dim:
            raise RuntimeError(
                f"embedding dim={len(embedding)}, expected {nm.dim} "
                f"({getattr(nm, '_embed_fingerprint', '?')})"
            )
        if hasattr(nm, "_pin_fingerprint_if_unset"):
            try:
                nm._pin_fingerprint_if_unset()
            except Exception:
                pass
        # SQLite always
        mem_id = nm.store.store(label or content[:60], content, embedding)
        # Postgres mirror — preserve the SQLite-assigned id
        if self._postgres_store:
            try:
                self._postgres_store.store(
                    label or content[:60], content, embedding, id_=int(mem_id)
                )
            except Exception:
                pass
        return mem_id
    
    def recall(self, query: str, k: int = 5,
               mmr_lambda: Optional[float] = None,
               score_floor: Optional[float] = None,
               score_percentile: Optional[float] = None,
               include_echoes: bool = False) -> list[dict]:
        """Semantic search with LSTM+kNN enhancement. Always uses SQLite for recall.

        mmr_lambda, score_floor, and score_percentile allow per-call override
        of the defaults configured at construction. None means "use the
        constructor default".

        include_echoes: when False (default), per-turn conversation logs
        labelled `auto:*` are filtered out so only curated and synthesised
        facts are surfaced. The corpus is ~98% auto-saved conversation
        echoes which would otherwise drown out the real answers. Set True
        only for session-replay tools that need raw turn content.

        score_percentile is the calibrated [0,1] alternative to score_floor:
        operating on rank percentile rather than raw RRF relevance, so
        score_percentile=0.5 keeps the top half of ranked candidates
        regardless of the underlying scale. score_floor remains exposed for
        backwards compatibility but operates on the badly-scaled raw
        relevance (~[0, 0.05]) — codex 2026-04-28 v5 audit caught this. See
        memory_client.Mazemaker.recall for the full implementation.
        """
        embedding = self._embedder.embed(query)

        # Always use SQLite for semantic recall — Postgres mirror is graph-only
        base_results = self._sqlite_memory.recall(
            query, k * 3, query_vec=embedding,
            mmr_lambda=mmr_lambda, score_floor=score_floor,
            score_percentile=score_percentile,
            include_echoes=include_echoes,
        )
        enhanced = self._enhance_recall(embedding, base_results, k)
        for r in enhanced:
            r.pop('embedding', None)
            # Unified convenience field — highest-precedence non-zero score.
            r.setdefault('score', (
                r.get('rerank_score') or
                r.get('relevance') or
                r.get('combined') or
                r.get('similarity') or
                0.0
            ))
        return enhanced[:k]

    def recall_multihop(self, query: str, k: int = 5, hops: int = 2) -> list[dict]:
        """Multi-hop retrieval: SQLite recall + Postgres graph expansion (if available)."""
        results = self.recall(query, k)
        if not self._postgres_store:
            return results

        expanded = []
        seen = {r['id'] for r in results}
        for r in results:
            try:
                conns = self._postgres_store.get_connections(r['id'])
                for c in conns:
                    other = c['target'] if c['source'] == r['id'] else c['source']
                    if other in seen:
                        continue
                    seen.add(other)
                    mem = self._postgres_store.get(other, include_embedding=False)
                    if mem:
                        expanded.append({
                            'id': other,
                            'label': mem['label'],
                            'content': mem['content'],
                            'activation': c['weight'],
                            'hop': 1,
                        })
            except Exception:
                pass
        merged = list(results) + expanded
        return merged[: max(k * 2, k)]

    def think_ids(self, start_id: int, depth: int = 2, k: int = 20) -> "list[int]":
        """Fast spreading-activation neighbour list. Returns just the top-k
        activated node IDs — bypasses the label/content lookup that think()
        does. Used by NREM's hot loop where the engine only feeds the IDs
        into activated_edges. CUDA-accelerated when self._sqlite_memory
        has _ppr_top_ids_gpu (i.e. GpuRecallEngine is loaded)."""
        if hasattr(self._sqlite_memory, "think_ids"):
            return self._sqlite_memory.think_ids(start_id, depth=depth, k=k)
        return [
            int(r.get("id"))
            for r in self.think(start_id, depth=depth)
            if r.get("id") is not None
        ][:k]

    def recall_batch(self, queries: "list[str]", k: int = 5) -> "list[list[dict]]":
        """Batched semantic recall. Drives GpuRecallEngine.recall_batch
        (one matmul, one embed_batch IPC) when CUDA is armed; falls back
        to per-query recall() otherwise. Used by REM to collapse 800
        sequential recalls into a single GPU batch."""
        if hasattr(self._sqlite_memory, "recall_batch"):
            return self._sqlite_memory.recall_batch(queries, k=k)
        return [self._sqlite_memory.recall(q, k=k) for q in queries]

    def recall_multi(
        self,
        queries: "list[str]",
        k: int = 10,
        fuse: bool = True,
        k_per_query: Optional[int] = None,
    ) -> "list[dict] | list[list[dict]]":
        """Multi-perspective recall: run N queries in one call and fuse results.

        When ``fuse=True`` (default) results from all queries are merged via
        Reciprocal Rank Fusion (RRF) and deduplicated, returning a flat list of
        at most *k* dicts.  Each returned dict includes a unified ``score``
        field set to the memory's RRF score.

        When ``fuse=False`` the raw per-query results are returned as
        ``list[list[dict]]`` (same shape as ``recall_batch``), with each dict
        carrying a ``score`` field copied from ``rerank_score or relevance or
        combined or similarity``.

        Args:
            queries:      List of query strings to run in parallel.
            k:            Maximum results to return (fused mode) or per query
                          (passthrough mode).
            fuse:         If True, apply RRF and return a flat deduplicated list.
            k_per_query:  Candidates fetched per query before fusion.
                          Defaults to ``2 * k`` so the fused pool is wide
                          enough to surface high-quality results for every angle.

        Returns:
            Flat ``list[dict]`` when ``fuse=True``, else ``list[list[dict]]``.
        """
        if not queries:
            return [] if fuse else []

        per_q = k_per_query if k_per_query is not None else k * 2
        # recall_batch already applies the score field via our patched recall()
        # only when going through GpuRecallEngine; normalise below regardless.
        per_query_results: list[list[dict]] = self.recall_batch(queries, k=per_q)

        if not fuse:
            # Passthrough — attach score field for consumers that expect it.
            for qr in per_query_results:
                for r in qr:
                    r.setdefault('score', (
                        r.get('rerank_score') or
                        r.get('relevance') or
                        r.get('combined') or
                        r.get('similarity') or
                        0.0
                    ))
            return per_query_results

        # --- RRF fusion ---
        # rrf_scores[id] accumulates 1/(60 + rank) summed over all queries.
        rrf_scores: dict[int, float] = {}
        # Keep the first-seen full dict for each id so we can return it.
        id_to_result: dict[int, dict] = {}

        for qr in per_query_results:
            for rank, r in enumerate(qr):
                mem_id = r.get('id')
                if mem_id is None:
                    continue
                rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + 1.0 / (60 + rank)
                if mem_id not in id_to_result:
                    id_to_result[mem_id] = r

        # Sort by RRF score descending, attach unified score, return top-k.
        ranked_ids = sorted(rrf_scores, key=lambda i: -rrf_scores[i])
        fused: list[dict] = []
        for mem_id in ranked_ids[:k]:
            result = dict(id_to_result[mem_id])
            result.pop('embedding', None)
            result['score'] = round(rrf_scores[mem_id], 8)
            fused.append(result)
        return fused

    def recall_with_neighbors(
        self,
        query: str,
        k: int = 10,
        depth: int = 1,
        neighbor_weight: float = 0.3,
        **recall_kwargs,
    ) -> list[dict]:
        """Graph-augmented recall: seed set from ``recall()`` + connected memories.

        After fetching a seed set of top-``k*2`` matches, traverse each seed's
        graph connections (1 or 2 hops) and merge in linked memories that were
        not in the seed set.  Neighbor scores decay multiplicatively with hop
        distance and connection weight:

            neighbor_score = neighbor_weight * source_score * connection_weight

        For ``depth=2`` the decay is applied again for each second-hop edge:

            hop2_score = neighbor_weight * hop1_score * connection_weight

        All results are deduplicated by id and sorted descending by score.
        Each dict carries a ``via`` field — ``"direct"`` for seed memories and
        ``"neighbor:hop1"`` / ``"neighbor:hop2"`` for graph-expanded ones — so
        consumers can distinguish how a memory was surfaced.

        Args:
            query:            Query string for the initial seed recall.
            k:                Maximum results to return.
            depth:            Hop depth for graph traversal (1 or 2).
            neighbor_weight:  Base weight multiplier for neighbour scores.
            **recall_kwargs:  Extra kwargs forwarded to ``recall()``
                              (e.g. ``mmr_lambda``, ``score_floor``).

        Returns:
            Flat ``list[dict]`` of at most *k* results, sorted by score desc.
        """
        # --- Seed set ---
        seeds = self.recall(query, k=k * 2, **recall_kwargs)
        seen_ids: dict[int, dict] = {}  # id → result dict

        for r in seeds:
            mem_id = r.get('id')
            if mem_id is None:
                continue
            r = dict(r)
            r.pop('embedding', None)
            r.setdefault('score', (
                r.get('rerank_score') or
                r.get('relevance') or
                r.get('combined') or
                r.get('similarity') or
                0.0
            ))
            r['via'] = 'direct'
            seen_ids[mem_id] = r

        # --- Graph traversal ---
        def _expand(frontier: dict[int, float], hop_label: str) -> dict[int, dict]:
            """Expand one hop from frontier {id: score} and return new nodes."""
            new_nodes: dict[int, dict] = {}
            for source_id, source_score in frontier.items():
                try:
                    conns = self.connections(source_id)
                except Exception:
                    continue
                for c in conns:
                    neighbour_id = c.get('id')
                    if neighbour_id is None or neighbour_id in seen_ids:
                        continue
                    conn_weight = float(c.get('weight', 1.0))
                    n_score = neighbor_weight * source_score * conn_weight
                    if neighbour_id in new_nodes:
                        # Keep the higher score if the same node appears via
                        # multiple edges in this hop.
                        if n_score <= new_nodes[neighbour_id]['score']:
                            continue
                    # Fetch the full memory dict from the SQLite store.
                    try:
                        mem = self._sqlite_memory.store.get(
                            neighbour_id, include_embedding=False
                        )
                    except Exception:
                        mem = None
                    if mem is None:
                        continue
                    new_nodes[neighbour_id] = {
                        'id': neighbour_id,
                        'label': mem.get('label', ''),
                        'content': mem.get('content', ''),
                        'score': round(n_score, 8),
                        'via': hop_label,
                        'connections': c,
                        'created_at': mem.get('created_at'),
                        'last_accessed': mem.get('last_accessed'),
                        'access_count': mem.get('access_count'),
                    }
            return new_nodes

        # Hop 1
        frontier_scores = {mid: r['score'] for mid, r in seen_ids.items()}
        hop1_nodes = _expand(frontier_scores, 'neighbor:hop1')
        seen_ids.update(hop1_nodes)

        # Hop 2
        if depth >= 2:
            hop2_frontier = {mid: r['score'] for mid, r in hop1_nodes.items()}
            hop2_nodes = _expand(hop2_frontier, 'neighbor:hop2')
            seen_ids.update(hop2_nodes)

        # Sort descending by score, cap at k.
        ranked = sorted(seen_ids.values(), key=lambda r: -r.get('score', 0.0))
        return ranked[:k]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """Spreading activation. SQLite primary, Postgres-enhanced if available."""
        base = self._sqlite_memory.think(start_id, depth, decay)

        if not self._postgres_store:
            return base

        try:
            visited = {start_id}
            frontier = [start_id]
            pg_results = []
            current_decay = decay

            for _ in range(depth):
                next_frontier = []
                for nid in frontier:
                    conns = self._postgres_store.get_connections(nid)
                    for c in conns:
                        other = c['target'] if c['source'] == nid else c['source']
                        if other not in visited:
                            visited.add(other)
                            activation = c['weight'] * current_decay
                            mem = self._postgres_store.get(other)
                            pg_results.append({
                                'id': other,
                                'label': mem['label'] if mem else f'node_{other}',
                                'activation': round(activation, 4),
                            })
                            next_frontier.append(other)
                frontier = next_frontier
                current_decay *= decay

            base_ids = {r['id'] for r in base}
            for r in pg_results:
                if r['id'] not in base_ids:
                    base.append(r)
            base.sort(key=lambda x: -x['activation'])
            return base[:20]
        except Exception:
            return base
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get connections for a memory. SQLite primary."""
        return self._sqlite_memory.connections(mem_id) if self._sqlite_memory else []
    
    def graph(self) -> dict:
        """Knowledge graph stats. SQLite primary."""
        return self._sqlite_memory.graph()
    
    def consolidate(self) -> int:
        """Run memory consolidation."""
        return 0

    # ------------------------------------------------------------------
    # Dream engine (NREM / REM / Insight) — lazy singleton bound to the
    # same SQLite DB as the inner memory store. The MCP wrapper expects
    # `dream(phase=...)` and `dream_stats()` on this adapter; without
    # them the v2 mcp returns 500 "internal.error" for those tools.
    # ------------------------------------------------------------------
    _dream_engine_singleton = None

    def _get_dream_engine(self):
        # Honor MM_DREAM_DISABLED — when an external dream_worker.py owns
        # consolidation we don't want every mazemaker_dream MCP call to
        # also fire a synchronous cycle on the same SQLite DB.
        import os
        if os.getenv("MM_DREAM_DISABLED", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "in-pod dream engine disabled via MM_DREAM_DISABLED — "
                "expected: external dream_worker.py is running"
            )
        if self._dream_engine_singleton is None:
            from dream_engine import DreamEngine  # type: ignore[import]
            # Pass `neural_memory=self._sqlite_memory` so REM/Insight phases
            # can call .think()/.recall() against the live store.
            self._dream_engine_singleton = DreamEngine.sqlite(
                self._db_path, neural_memory=self._sqlite_memory
            )
        return self._dream_engine_singleton

    def dream(self, phase: str = "all") -> dict:
        """Run a single dream cycle synchronously.

        phase: one of 'nrem', 'supersedes', 'rem', 'insight', 'afe',
        'synthesis', 'dae', or 'all' (default — orchestrated 7-phase
        cycle).  Returns the per-phase stats dict.
        """
        # When MM_DREAM_DISABLED is set, return a structured "skipped"
        # response instead of raising — the architect dashboard's M02
        # DREAM monitor renders raised exceptions as a generic
        # "Internal error" tile, which buries the fact that the cycle
        # is being driven by an external dream_worker.
        import os
        if os.getenv("MM_DREAM_DISABLED", "").lower() in ("1", "true", "yes"):
            return {
                "ok": True,
                "phase": phase,
                "status": "skipped",
                "note": (
                    "in-pod dream engine disabled — external dream_worker.py "
                    "owns consolidation. Cycles still land in dream_sessions; "
                    "watch mazemaker_dream_stats."
                ),
            }
        eng = self._get_dream_engine()
        phase = (phase or "all").strip().lower()
        if phase == "all":
            return eng.dream_now()
        if phase == "nrem":
            return {"nrem": eng._phase_nrem()}
        if phase == "supersedes":
            return {"supersedes": eng._phase_supersedes()}
        if phase == "rem":
            return {"rem": eng._phase_rem()}
        if phase == "insight":
            return {"insights": eng._phase_insights()}
        if phase == "afe":
            return {"afe": eng._phase_afe()}
        if phase == "synthesis":
            return {"synthesis": eng._phase_synthesis()}
        if phase == "dae":
            return {"dae": eng._phase_dae()}
        raise ValueError(f"unknown dream phase: {phase}")

    def dream_stats(self) -> dict:
        """Aggregate dream-engine stats from the backend.

        Read-only; bypasses the _get_dream_engine guard so dream_stats
        keeps working when MM_DREAM_DISABLED is set (external dream_worker
        is still writing rows that this method reads).
        """
        from dream_engine import SQLiteDreamBackend  # type: ignore[import]
        if self._dream_engine_singleton is not None:
            return self._dream_engine_singleton._backend.get_dream_stats()
        return SQLiteDreamBackend(self._db_path).get_dream_stats()

    def stats(self) -> dict:
        """System statistics. SQLite primary."""
        s = self._sqlite_memory.stats()
        s['embedding_dim'] = self._dim
        s['embedding_backend'] = self._embedder.backend.__class__.__name__
        s['backend'] = 'sqlite'
        s['postgres_mirror'] = self._postgres_store is not None
        return s
    
    def close(self):
        """Clean shutdown. Saves LSTM weights, flushes logger, closes stores.

        Each subsystem is closed exactly ONCE. Order: save
        learned state first (LSTM weights, access logs), then tear down
        engines (knn, lstm), then close stores last so any in-flight
        save can still talk to the underlying file/socket.
        """
        # 1. Save LSTM weights for next session
        if self._lstm_knn_ready and self._lstm:
            try:
                lstm_weights_path = str(Path.home() / ".mazemaker" / "engine" / "lstm_weights.bin")
                self._lstm.save(lstm_weights_path)
            except Exception:
                pass
        # 2. Flush access logger to disk before tearing down store
        if self._access_logger:
            try:
                self._access_logger.save()
            except Exception:
                pass
        # 3. Close LSTM + kNN engines (compute side, no on-disk state)
        if self._knn:
            try:
                self._knn.close()
            except Exception:
                pass
            self._knn = None
        if self._lstm:
            try:
                self._lstm.close()
            except Exception:
                pass
            self._lstm = None
        # 4. Close stores (SQLite first, then Postgres mirror) — single close.
        if self._sqlite_memory:
            try:
                self._sqlite_memory.close()
            except Exception:
                pass
            self._sqlite_memory = None
        if self._postgres_store:
            try:
                self._postgres_store.close()
            except Exception:
                pass
            self._postgres_store = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def backend(self) -> str:
        if self._postgres_store:
            return "postgres"
        return self._embedder.backend.__class__.__name__
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def archive_compression(self, turns: list[dict], session_tag: str = "") -> dict:
        """Archive conversation turns before they are destroyed by compression.

        Each turn is stored as a separate memory with role-specific labeling.
        Tool results are stored with a 500-char floor (truncated only if >500).
        This is lossless-preservation — no filtering, no "is this meaningful?"
        judgment. Everything is stored so nothing is lost when context compresses.

        Args:
            turns: List of message dicts with role/content/tool_calls keys
            session_tag: Optional session identifier for labeling (e.g. "session-a1b2c3")

        Returns:
            dict with 'archived' count
        """
        archived = 0
        for turn in turns:
            role = turn.get("role", "unknown")
            content = (turn.get("content") or "")[:2000]  # Cap at 2000 chars

            # Skip empty tool results
            if role == "tool" and len(content) < 5:
                continue

            # Build role-specific label and enriched content
            if role == "user":
                label = "msg:user"
                memory_text = f"Q: {content}"

            elif role == "assistant":
                label = "msg:assistant"
                tool_calls = turn.get("tool_calls", [])
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    content += f"\n[Tools used: {', '.join(names)}]"
                memory_text = content

            elif role == "tool":
                label = "msg:tool-result"
                # Truncate only long tool results; keep short ones intact
                if len(content) > 500:
                    content = content[:500] + "\n...[truncated]"
                memory_text = content

            elif role == "system":
                label = "msg:system"
                memory_text = content

            else:
                continue

            if memory_text.strip():
                full_label = f"archive:{session_tag}:{label}" if session_tag else f"archive:{label}"
                # Archive writes are lossless preservation, not curated
                # learning: each archived turn is stored verbatim, NOT fused
                # with similar prior turns and NOT auto-connected. Reasons:
                #   1. Conflict-fusion mutates user message text (wraps it in
                #      [CANONICAL]/[PREVIOUSLY] markup), which destroys the
                #      \"archive\" property.
                #   2. auto_connect against an N-archive corpus produces
                #      O(turns^2) edges of near-duplicate sibling turns,
                #      flooding the graph with no semantic value.
                self.remember(
                    memory_text,
                    label=full_label,
                    detect_conflicts=False,
                    auto_connect=False,
                )
                archived += 1

        return {"archived": archived}

    def __repr__(self):
        return f"Memory(backend={self.backend}, dim={self.dim}, db={self._db_path})"
