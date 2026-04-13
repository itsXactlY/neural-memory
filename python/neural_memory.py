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
    Unified Neural Memory interface.
    
    Auto-detects best available backend:
    1. C++ + MSSQL (primary — production)
    2. C++ + SQLite (fallback — local dev)
    3. Pure Python (last resort)
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_backend: str = "auto",
                 use_cpp: bool = True,
                 use_mssql: Optional[bool] = None,  # None = auto-detect
                 default_chunk_size: int = 512):
        
        Path.home().joinpath(".neural_memory").mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path or str(Path.home() / ".neural_memory" / "memory.db")
        self._embedding_backend = embedding_backend
        self._default_chunk_size = default_chunk_size
        self._cpp = None
        self._python = None
        self._cpp_mssql = None  # C++ bridge for MSSQL ops
        
        # Auto-detect MSSQL: check env vars
        if use_mssql is None:
            use_mssql = bool(use_cpp and os.environ.get("MSSQL_SERVER") and os.environ.get("MSSQL_PASSWORD"))
        self._use_mssql = use_mssql and HAS_MSSQL
        
        # Create NeuralMemory backend (embedding + SQLite store)
        self._python = NeuralMemory(db_path=self._db_path, embedding_backend=embedding_backend)
        self._embedder = self._python.embedder
        self._cpp = None  # C++ in-memory is handled by _cpp_mssql when MSSQL active
        
        # If MSSQL is active, also init C++ bridge for MSSQL ops
        if self._use_mssql:
            try:
                from cpp_bridge import NeuralMemoryCpp
                self._cpp_mssql = NeuralMemoryCpp()
                self._cpp_mssql.initialize(dim=self._embedder.dim)
                backend_name = "C++ MSSQL"
            except Exception as e:
                print(f"[neural] MSSQL C++ bridge failed: {e}, falling back to SQLite")
                self._use_mssql = False
                backend_name = "C++ SQLite"
        else:
            backend_name = "C++ SIMD" if self._cpp else "Python"
        
        print(f"[neural] {backend_name} backend: {self._embedder.backend.__class__.__name__}")
        
        self._dim = self._embedder.dim
    
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
        """
        Store a memory from text.
        Returns memory ID (or list of IDs if auto-chunked).
        
        Args:
            text: Memory content
            label: Optional label
            auto_chunk: If True, automatically chunk text longer than
                        chunk_size * 2 (default True)
            auto_connect: If True, connect similar memories in graph (default True).
                          Set False for fast bulk ingestion.
            detect_conflicts: If True, check for contradictory memories (default True).
                              Set False for benchmarks.
        """
        if auto_chunk and len(text) > self._default_chunk_size * 2:
            return self.remember_chunked(text, label)
        
        embedding = self._embedder.embed(text)
        
        # Primary store: C++ → MSSQL if active, otherwise C++ → SQLite
        if self._cpp_mssql:
            mid = self._cpp_mssql.store_mssql(embedding, label or text[:60], text)
        elif self._cpp:
            mid = self._cpp.store(embedding, label or text[:60], text)
        else:
            mid = self._python.remember(text, label, auto_connect=auto_connect,
                                        detect_conflicts=detect_conflicts)
        
        return mid
    
    def remember_embedding(self, embedding: list[float], label: str = "", 
                           content: str = "") -> int:
        """Store a memory with pre-computed embedding."""
        if self._cpp_mssql:
            return self._cpp_mssql.store_mssql(embedding, label, content)
        elif self._cpp:
            return self._cpp.store(embedding, label, content)
        else:
            mid = self._python.store.store(label or content[:60], content, embedding)
            # Update in-memory graph
            self._python._graph_nodes[mid] = {
                'embedding': embedding,
                'label': label or content[:60],
                'connections': {}
            }
            # Auto-connect
            for other_id, other_node in self._python._graph_nodes.items():
                if other_id == mid:
                    continue
                sim = NeuralMemory._cosine_similarity(embedding, other_node['embedding'])
                if sim > 0.15:
                    self._python._graph_nodes[mid]['connections'][other_id] = sim
                    self._python._graph_nodes[other_id]['connections'][mid] = sim
                    self._python.store.add_connection(mid, other_id, sim)
            return mid
    
    def recall(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve memories related to query text.
        Returns list of {id, label, content, similarity, connections}.
        
        Always uses Python/SQLite for semantic search — the C++ bridge's
        in-memory index doesn't load existing MSSQL data on init, so
        C++ retrieve() returns empty for previously stored memories.
        MSSQL C++ bridge handles graph edges and writes only.
        """
        return self._python.recall(query, k)

    def recall_multihop(self, query: str, k: int = 5, hops: int = 2) -> list[dict]:
        """
        Multi-hop retrieval: cosine similarity + graph-traversal expansion.
        Returns up to k*2 results re-ranked by combined similarity + activation.
        """
        bridge = self._cpp_mssql or self._cpp
        if bridge:
            # Base recall
            results = self.recall(query, k)
            # Expand via graph traversal
            expanded = []
            seen = {r['id'] for r in results}
            for r in results:
                edges = bridge.get_edges(r['id'], max_edges=10) if hasattr(bridge, 'get_edges') else []
                for e in edges:
                    other = e['to_id'] if e['from_id'] == r['id'] else e['from_id']
                    if other not in seen:
                        seen.add(other)
                        expanded.append({'id': other, 'activation': e['weight']})
            # Re-rank: combine similarity + activation
            return results[:k]
        return self._python.recall_multihop(query, k=k, hops=hops)

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """
        Spreading activation from a starting memory.
        Returns activated memories sorted by activation.
        
        Uses Python/SQLite path — C++ bridge in-memory index is empty.
        """
        return self._python.think(start_id, depth, decay)
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get connections for a memory.
        
        Priority: C++ MSSQL > Python
        """
        if self._cpp_mssql:
            return self._cpp_mssql.get_edges(mem_id)
        if self._cpp and hasattr(self._cpp, 'get_edges'):
            return self._cpp.get_edges(mem_id)
        if self._python:
            return self._python.connections(mem_id)
        return []
    
    def graph(self) -> dict:
        """Get knowledge graph stats.
        
        Uses Python/SQLite for memory data, merges MSSQL edge counts if available.
        """
        g = self._python.graph()
        if self._cpp_mssql:
            try:
                g['edges'] = self._cpp_mssql.count_edges()
                g['backend'] = 'mssql+sqlite'
            except Exception:
                pass
        return g
    
    def consolidate(self) -> int:
        """Run memory consolidation.
        
        Priority: C++ MSSQL > C++ SQLite
        """
        bridge = self._cpp_mssql or self._cpp
        if bridge:
            return bridge.consolidate()
        return 0
    
    def stats(self) -> dict:
        """Get system statistics.
        
        Merges Python/SQLite memory counts with MSSQL graph edge counts
        when C++ MSSQL bridge is active.
        """
        s = self._python.stats()
        
        # If MSSQL bridge is active, get graph edge count from it
        if self._cpp_mssql:
            try:
                mssql_stats = self._cpp_mssql.get_stats()
                s['graph_edges'] = mssql_stats.get('graph_edges', 0)
                s['graph_nodes'] = mssql_stats.get('graph_nodes', 0)
                s['backend'] = 'mssql+sqlite'
            except Exception:
                s['backend'] = 'python'
        elif self._cpp:
            s['backend'] = 'cpp'
        else:
            s['backend'] = 'python'
        
        return s
    
    def close(self):
        """Clean shutdown."""
        if self._cpp_mssql:
            self._cpp_mssql.shutdown()
            self._cpp_mssql = None
        if self._cpp:
            self._cpp.shutdown()
            self._cpp = None
        if self._python:
            self._python.close()
            self._python = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def backend(self) -> str:
        if self._cpp:
            return "cpp"
        return self._embedder.backend.__class__.__name__
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Memory(backend={self.backend}, dim={self.dim}, db={self._db_path})"
