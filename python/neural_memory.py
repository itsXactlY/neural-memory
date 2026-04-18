#!/usr/bin/env python3
"""
neural_memory.py - THE Unified Neural Memory API
One import to rule them all.

Architecture:
  C++ MSSQL (primary) ─── GraphNodes/GraphEdges/NeuralMemory tables
  C++ SQLite (fallback) ── memory.db local cache
  Python ──────────────── Dream engine, embedding, orchestration

Usage:
    from neural_memory import Memory
    
    mem = Memory()  # Auto-detects MSSQL vs SQLite
    mem.remember("The user has a dog named Lou")
    results = mem.recall("What pet does the user have?")
    mem.think(results[0].id)
    mem.consolidate()
    mem.close()
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Add python dir to path
sys.path.insert(0, str(Path(__file__).parent))

from embed_provider import EmbeddingProvider
from memory_client import NeuralMemory, SQLiteStore

# Try MSSQL, fall back to SQLite
try:
    from mssql_store import MSSQLStore
    HAS_MSSQL = True
except ImportError:
    HAS_MSSQL = False


class Memory:
    """
    Unified Neural Memory interface with LSTM+kNN enhancement.
    
    Backend priority:
    1. MSSQL (via pyodbc) — when MSSQL is installed and running
    2. SQLite (via Python) — fallback when MSSQL unavailable
    
    LSTM+kNN is auto-initialized when libneural_memory.so is available.
    - AccessLogger: records every recall event
    - LSTMPredictor: learns access patterns, predicts next relevant embedding
    - KNNEngine: multi-signal re-ranking (embedding + temporal + frequency + graph)
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_backend: str = "auto",
                 use_cpp: bool = True,
                 use_mssql: Optional[bool] = None,
                 default_chunk_size: int = 512):
        
        Path.home().joinpath(".neural_memory").mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path or str(Path.home() / ".neural_memory" / "memory.db")
        self._default_chunk_size = default_chunk_size
        self._mssql_store = None
        self._sqlite_memory = None
        
        # Embedder (shared regardless of backend)
        from embed_provider import EmbeddingProvider
        self._embedder = EmbeddingProvider(backend=embedding_backend)
        self._dim = self._embedder.dim
        
        # Auto-detect MSSQL
        if use_mssql is None:
            use_mssql = bool(os.environ.get("MSSQL_SERVER") and os.environ.get("MSSQL_PASSWORD"))
        
        # Try MSSQL first
        if use_mssql:
            try:
                from mssql_store import MSSQLStore
                self._mssql_store = MSSQLStore()
                print(f"[neural] MSSQL backend: {self._embedder.backend.__class__.__name__} ({self._dim}d)")
            except Exception as e:
                print(f"[neural] MSSQL unavailable ({e}), falling back to SQLite")
                use_mssql = False

        # SQLite always needed for semantic recall (MSSQLStore has no recall method)
        from memory_client import NeuralMemory
        self._sqlite_memory = NeuralMemory(db_path=self._db_path, embedding_backend=embedding_backend)
        if not use_mssql:
            print(f"[neural] SQLite backend: {self._embedder.backend.__class__.__name__} ({self._dim}d)")
        else:
            print(f"[neural] Hybrid mode: MSSQL (graph) + SQLite (recall)")
        
        # --- LSTM + kNN (auto-initialized) ---
        self._access_logger = None
        self._lstm = None
        self._knn = None
        self._lstm_knn_ready = False
        self._init_lstm_knn()
    
    def _init_lstm_knn(self):
        """Auto-initialize LSTM+kNN if libneural_memory.so is available."""
        try:
            from access_logger import AccessLogger
            from lstm_knn_bridge import LSTMPredictor, KNNEngine
            
            self._access_logger = AccessLogger.instance()
            self._lstm = LSTMPredictor(input_dim=self._dim, hidden_dim=256)
            self._knn = KNNEngine(embed_dim=self._dim)
            
            # Try to load saved LSTM weights
            lstm_weights_path = Path.home() / ".neural_memory" / "lstm_weights.bin"
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
            
            # 2. LSTM prediction from access sequence
            lstm_context = None
            recent = self._access_logger.get_sequence(n=15)
            if len(recent) >= 3:
                try:
                    seq_embs = [e["query_emb"] for e in recent[-10:]]
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
                
                # If embedding not in result, try to fetch it
                if emb is None:
                    if self._mssql_store:
                        try:
                            full = self._mssql_store.get(mem_id)
                            emb = full.get('embedding', []) if full else []
                        except Exception:
                            emb = []
                    elif self._sqlite_memory:
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
            
            # 6. Train LSTM in background (fire-and-forget)
            try:
                if len(recent) >= 2 and lstm_context is not None:
                    # Training pair: sequence → actual query embedding
                    seq_embs = [e["query_emb"] for e in recent[-10:]]
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
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                # Take last `overlap` chars, but start at a word boundary
                overlap_text = prev[-overlap:]
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                overlapped.append(overlap_text + " " + chunks[i])
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
        
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        if len(chunks) == 1:
            return [self.remember(chunks[0], label, auto_chunk=False)]
        
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_label = f"{label} [chunk {i+1}/{len(chunks)}]" if label else f"[chunk {i+1}/{len(chunks)}]"
            ids.append(self.remember(chunk, chunk_label, auto_chunk=False))
        
        return ids
    
    def remember(self, text: str, label: str = "", auto_chunk: bool = True,
                 auto_connect: bool = True, detect_conflicts: bool = True) -> int | list[int]:
        """Store a memory. SQLite primary, MSSQL mirror. Returns memory ID."""
        if auto_chunk and len(text) > self._default_chunk_size * 2:
            return self.remember_chunked(text, label)
        
        embedding = self._embedder.embed(text)
        
        # SQLite always (source of truth)
        mem_id = self._sqlite_memory.remember(text, label, auto_connect=auto_connect,
                                              detect_conflicts=detect_conflicts)
        # MSSQL mirror (optional backup)
        if self._mssql_store:
            try:
                self._mssql_store.store(label or text[:60], text, embedding)
            except Exception:
                pass
        return mem_id
    
    def remember_embedding(self, embedding: list[float], label: str = "", 
                           content: str = "") -> int:
        """Store a memory with pre-computed embedding. SQLite primary, MSSQL mirror."""
        # SQLite always
        mem_id = self._sqlite_memory.store.store(label or content[:60], content, embedding)
        # MSSQL mirror
        if self._mssql_store:
            try:
                self._mssql_store.store(label or content[:60], content, embedding)
            except Exception:
                pass
        return mem_id
    
    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search with LSTM+kNN enhancement. Always uses SQLite for recall (MSSQLStore has no recall)."""
        embedding = self._embedder.embed(query)

        # Always use SQLite for semantic recall — MSSQL is graph-only
        base_results = self._sqlite_memory.recall(query, k * 3)
        enhanced = self._enhance_recall(embedding, base_results, k)
        for r in enhanced:
            r.pop('embedding', None)
        return enhanced[:k]

    def recall_multihop(self, query: str, k: int = 5, hops: int = 2) -> list[dict]:
        """Multi-hop retrieval: SQLite recall + MSSQL graph expansion (if available)."""
        results = self.recall(query, k)
        if not self._mssql_store:
            return results
        
        expanded = []
        seen = {r['id'] for r in results}
        for r in results:
            try:
                conns = self._mssql_store.get_connections(r['id'])
                for c in conns:
                    other = c['target'] if c['source'] == r['id'] else c['source']
                    if other not in seen:
                        seen.add(other)
                        mem = self._mssql_store.get(other)
                        if mem:
                            expanded.append({'id': other, 'label': mem['label'],
                                            'content': mem['content'], 'activation': c['weight']})
            except Exception:
                pass
        return results[:k]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """Spreading activation. SQLite primary, MSSQL enhanced if available."""
        # Always use SQLite think
        base = self._sqlite_memory.think(start_id, depth, decay)
        
        if not self._mssql_store:
            return base
        
        # MSSQL graph expansion (enhances with additional connections)
        try:
            visited = {start_id}
            frontier = [start_id]
            mssql_results = []
            current_decay = decay
            
            for _ in range(depth):
                next_frontier = []
                for nid in frontier:
                    conns = self._mssql_store.get_connections(nid)
                    for c in conns:
                        other = c['target'] if c['source'] == nid else c['source']
                        if other not in visited:
                            visited.add(other)
                            activation = c['weight'] * current_decay
                            mem = self._mssql_store.get(other)
                            mssql_results.append({
                                'id': other,
                                'label': mem['label'] if mem else f'node_{other}',
                                'activation': round(activation, 4),
                            })
                            next_frontier.append(other)
                frontier = next_frontier
                current_decay *= decay
            
            # Merge: SQLite results first, MSSQL adds depth
            base_ids = {r['id'] for r in base}
            for r in mssql_results:
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
    
    def stats(self) -> dict:
        """System statistics. SQLite primary."""
        s = self._sqlite_memory.stats()
        s['embedding_dim'] = self._dim
        s['embedding_backend'] = self._embedder.backend.__class__.__name__
        s['backend'] = 'sqlite'
        s['mssql_mirror'] = self._mssql_store is not None
        return s
    
    def close(self):
        """Clean shutdown. Saves LSTM weights, closes stores."""
        # Save LSTM weights for next session
        if self._lstm_knn_ready and self._lstm:
            try:
                lstm_weights_path = str(Path.home() / ".neural_memory" / "lstm_weights.bin")
                self._lstm.save(lstm_weights_path)
            except Exception:
                pass
        # Close kNN engine
        if self._knn:
            try:
                self._knn.close()
            except Exception:
                pass
        # Close stores (SQLite first, then MSSQL mirror)
        try:
            self._sqlite_memory.close()
        except Exception:
            pass
        if self._mssql_store:
            try:
                self._mssql_store.close()
            except Exception:
                pass
        self._knn = None
        # Close LSTM
        if self._lstm:
            try:
                self._lstm.close()
            except Exception:
                pass
            self._lstm = None
        # Flush access logger
        if self._access_logger:
            try:
                self._access_logger.save()
            except Exception:
                pass
        if self._mssql_store:
            self._mssql_store.close()
            self._mssql_store = None
        if self._sqlite_memory:
            self._sqlite_memory.close()
            self._sqlite_memory = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def backend(self) -> str:
        if self._mssql_store:
            return "mssql"
        return self._embedder.backend.__class__.__name__
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Memory(backend={self.backend}, dim={self.dim}, db={self._db_path})"
