#!/usr/bin/env python3
"""
memory_client.py - Python client for Neural Memory Adapter
Wraps the C++ library via ctypes. Uses embed_provider for text->vector.
"""

import ctypes
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

# ============================================================================
# Find the shared library
# ============================================================================

def _find_lib():
    candidates = [
        Path(__file__).parent.parent / "build" / "libneural_memory.so",
        Path.home() / "projects" / "neural-memory-adapter" / "build" / "libneural_memory.so",
        Path("/usr/local/lib/libneural_memory.so"),
        Path("/usr/lib/libneural_memory.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("libneural_memory.so not found. Build first: cd build && cmake --build .")

# ============================================================================
# C API struct definitions
# ============================================================================

class CSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 1024),
    ]

class CStats(ctypes.Structure):
    _fields_ = [
        ("total_stores", ctypes.c_uint64),
        ("total_retrieves", ctypes.c_uint64),
        ("total_searches", ctypes.c_uint64),
        ("total_consolidations", ctypes.c_uint64),
        ("avg_store_us", ctypes.c_uint64),
        ("avg_retrieve_us", ctypes.c_uint64),
        ("graph_nodes", ctypes.c_size_t),
        ("graph_edges", ctypes.c_size_t),
        ("hopfield_patterns", ctypes.c_size_t),
        ("hopfield_occupancy", ctypes.c_float),
    ]

# ============================================================================
# SQLite persistence layer
# ============================================================================

DB_PATH = Path.home() / ".neural_memory" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT (unixepoch()),
    last_accessed REAL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    edge_type TEXT DEFAULT 'similar',
    created_at REAL DEFAULT (unixepoch()),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_id);
"""

class SQLiteStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        self.conn.execute("PRAGMA synchronous=NORMAL") # Faster writes
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._lock = threading.Lock()
    
    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                (label, content, blob)
            )
            self.conn.commit()
            return cur.lastrowid
    
    def get_all(self) -> list[dict]:
        import struct
        rows = self.conn.execute(
            "SELECT id, label, content, embedding, salience, access_count FROM memories ORDER BY id"
        ).fetchall()
        results = []
        for row in rows:
            id_, label, content, blob, salience, access_count = row
            dim = len(blob) // 4
            embedding = list(struct.unpack(f'{dim}f', blob))
            results.append({
                'id': id_, 'label': label, 'content': content,
                'embedding': embedding, 'salience': salience,
                'access_count': access_count
            })
        return results
    
    def get(self, id_: int) -> Optional[dict]:
        import struct
        row = self.conn.execute(
            "SELECT id, label, content, embedding, salience, access_count FROM memories WHERE id = ?",
            (id_,)
        ).fetchone()
        if not row:
            return None
        id_, label, content, blob, salience, access_count = row
        dim = len(blob) // 4
        embedding = list(struct.unpack(f'{dim}f', blob))
        return {
            'id': id_, 'label': label, 'content': content,
            'embedding': embedding, 'salience': salience,
            'access_count': access_count
        }
    
    def touch(self, id_: int):
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET last_accessed = unixepoch(), access_count = access_count + 1 WHERE id = ?",
                (id_,)
            )
            self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
                (source, target, weight, edge_type)
            )
            self.conn.commit()
    
    def get_connections(self, node_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT source_id, target_id, weight, edge_type FROM connections 
               WHERE source_id = ? OR target_id = ? ORDER BY weight DESC""",
            (node_id, node_id)
        ).fetchall()
        return [
            {'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3]}
            for r in rows
        ]
    
    def get_stats(self) -> dict:
        mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn_count = self.conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        return {'memories': mem_count, 'connections': conn_count}
    
    def close(self):
        self.conn.close()


# ============================================================================
# Neural Memory Client
# ============================================================================

class NeuralMemory:
    """
    Python interface to the Neural Memory system.
    
    Usage:
        mem = NeuralMemory()
        mem.remember("The user has a dog named Lou")
        mem.remember("Working on BTQuant trading platform")
        results = mem.recall("What pet does the user have?")
    """
    
    def __init__(self, db_path: str | Path = DB_PATH, embedding_backend: str = "auto"):
        from embed_provider import EmbeddingProvider
        
        self.embedder = EmbeddingProvider(backend=embedding_backend)
        self.store = SQLiteStore(db_path)
        self.dim = self.embedder.dim
        
        # In-memory graph for spreading activation
        self._graph_nodes: dict[int, dict] = {}  # id -> {embedding, connections}
        self._load_from_store()
    
    def _load_from_store(self):
        """Load existing memories into in-memory graph"""
        for mem in self.store.get_all():
            self._graph_nodes[mem['id']] = {
                'embedding': mem['embedding'],
                'label': mem['label'],
                'connections': {}
            }
        
        # Load connections
        for mem in self.store.get_all():
            conns = self.store.get_connections(mem['id'])
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                self._graph_nodes[mem['id']]['connections'][other] = c['weight']
    
    def remember(self, text: str, label: str = "") -> int:
        """Store a memory. Returns memory ID."""
        import math
        import time
        
        embedding = self.embedder.embed(text)
        mem_id = self.store.store(label or text[:60], text, embedding)
        
        # Add to in-memory graph
        self._graph_nodes[mem_id] = {
            'embedding': embedding,
            'label': label or text[:60],
            'connections': {}
        }
        
        # Auto-connect to similar memories
        for other_id, other_node in self._graph_nodes.items():
            if other_id == mem_id:
                continue
            sim = self._cosine_similarity(embedding, other_node['embedding'])
            if sim > 0.15:  # Threshold for connection
                self._graph_nodes[mem_id]['connections'][other_id] = sim
                self._graph_nodes[other_id]['connections'][mem_id] = sim
                self.store.add_connection(mem_id, other_id, sim)
        
        return mem_id
    
    def recall(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve memories related to query.
        Returns list of {id, label, content, similarity, connections}.
        """
        import math
        
        query_vec = self.embedder.embed(query)
        
        # Score all memories
        scored = []
        for mem in self.store.get_all():
            sim = self._cosine_similarity(query_vec, mem['embedding'])
            scored.append({**mem, 'similarity': sim})
        
        # Sort by similarity
        scored.sort(key=lambda x: -x['similarity'])
        
        # Enrich with connections via spreading activation
        results = []
        seen = set()
        for mem in scored[:k]:
            if mem['id'] in seen:
                continue
            seen.add(mem['id'])
            
            # Get connections
            conns = self.store.get_connections(mem['id'])
            connected = []
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                if other not in seen:
                    other_mem = self.store.get(other)
                    if other_mem:
                        connected.append({
                            'id': other,
                            'label': other_mem['label'],
                            'weight': c['weight']
                        })
            
            results.append({
                'id': mem['id'],
                'label': mem['label'],
                'content': mem['content'],
                'similarity': round(mem['similarity'], 4),
                'connections': connected[:3],  # Top 3
            })
            
            self.store.touch(mem['id'])
        
        return results
    
    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """
        Spreading activation from a starting memory.
        Returns activated memories sorted by activation level.
        """
        if start_id not in self._graph_nodes:
            return []
        
        # BFS with activation propagation
        activation = {start_id: 1.0}
        visited = {start_id}
        queue = [(start_id, 1.0, 0)]
        
        while queue:
            current, act, level = queue.pop(0)
            if level >= depth or act < 0.01:
                continue
            
            node = self._graph_nodes.get(current, {})
            for neighbor_id, weight in node.get('connections', {}).items():
                propagated = act * weight * decay
                if propagated < 0.01:
                    continue
                
                if neighbor_id not in activation or propagated > activation[neighbor_id]:
                    activation[neighbor_id] = propagated
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, propagated, level + 1))
        
        # Build results (skip start node)
        results = []
        for node_id, act in activation.items():
            if node_id == start_id:
                continue
            mem = self.store.get(node_id)
            if mem:
                results.append({
                    'id': node_id,
                    'label': mem['label'],
                    'activation': round(act, 4),
                })
        
        results.sort(key=lambda x: -x['activation'])
        return results
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get all connections for a memory."""
        conns = self.store.get_connections(mem_id)
        results = []
        for c in conns:
            other = c['target'] if c['source'] == mem_id else c['source']
            mem = self.store.get(other)
            if mem:
                results.append({
                    'id': other,
                    'label': mem['label'],
                    'weight': round(c['weight'], 4),
                    'type': c['type']
                })
        return results
    
    def graph(self) -> dict:
        """Get knowledge graph stats and structure."""
        stats = self.store.get_stats()
        
        # Build adjacency summary
        edges = []
        seen = set()
        for node_id, node in self._graph_nodes.items():
            for other_id, weight in node.get('connections', {}).items():
                key = tuple(sorted([node_id, other_id]))
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        'from': node_id,
                        'to': other_id,
                        'weight': round(weight, 3)
                    })
        
        return {
            'nodes': stats['memories'],
            'edges': len(edges),
            'top_edges': sorted(edges, key=lambda x: -x['weight'])[:10]
        }
    
    def stats(self) -> dict:
        """Get system statistics."""
        graph = self.store.get_stats()
        return {
            'memories': graph['memories'],
            'connections': graph['connections'],
            'embedding_dim': self.dim,
            'embedding_backend': self.embedder.backend.__class__.__name__,
        }
    
    def close(self):
        self.store.close()
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = (sum(x*x for x in a)) ** 0.5
        nb = (sum(x*x for x in b)) ** 0.5
        return dot / (na * nb) if na * nb > 1e-10 else 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
