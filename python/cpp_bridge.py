#!/usr/bin/env python3
"""
cpp_bridge.py - ctypes wrapper for the C++ Mazemaker library
Provides Pythonic interface to libmazemaker.so
"""

import ctypes
import ctypes.util
from pathlib import Path
from typing import Optional

# ============================================================================
# Find library
# ============================================================================

def _find_lib() -> str:
    candidates = [
        Path(__file__).parent.parent / "build" / "libmazemaker.so",
        Path.home() / "projects" / "mazemaker-adapter" / "build" / "libmazemaker.so",
        Path("/usr/local/lib/libmazemaker.so"),
        Path("/usr/lib/libmazemaker.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Try LD_LIBRARY_PATH
    lib = ctypes.util.find_library("neural_memory")
    if lib:
        return lib
    raise FileNotFoundError(
        "libmazemaker.so not found. Build first:\n"
        "  cd ~/projects/mazemaker-adapter/build && cmake --build . -j$(nproc)"
    )

# ============================================================================
# C API types (matching c_api.h)
# ============================================================================

class CSearchResult(ctypes.Structure):
    """Must match MazemakerResult in c_api.h exactly."""
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("embedding", ctypes.POINTER(ctypes.c_float)),  # float* — caller must NOT free
        ("embedding_dim", ctypes.c_int),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 4096),
        ("similarity", ctypes.c_float),
        ("salience", ctypes.c_float),
    ]

class CStats(ctypes.Structure):
    """Must match MazemakerStats in c_api.h exactly."""
    _fields_ = [
        ("episodic_count", ctypes.c_size_t),
        ("semantic_count", ctypes.c_size_t),
        ("episodic_occupancy", ctypes.c_float),
        ("semantic_occupancy", ctypes.c_float),
        ("hopfield_patterns", ctypes.c_size_t),
        ("hopfield_occupancy", ctypes.c_float),
        ("graph_nodes", ctypes.c_size_t),
        ("graph_edges", ctypes.c_size_t),
        ("graph_density", ctypes.c_float),
        # Note: ctypes will insert padding here for uint64_t alignment, matching C compiler behavior
        ("avg_store_us", ctypes.c_uint64),
        ("avg_retrieve_us", ctypes.c_uint64),
        ("avg_search_us", ctypes.c_uint64),
        ("total_stores", ctypes.c_uint64),
        ("total_retrieves", ctypes.c_uint64),
        ("total_searches", ctypes.c_uint64),
        ("total_consolidations", ctypes.c_uint64),
    ]

# ============================================================================
# C++ Bridge
# ============================================================================

class MazemakerCpp:
    """
    Pythonic wrapper for the C++ Mazemaker library.
    
    Usage:
        mem = MazemakerCpp()
        mem.initialize(dim=1024)
        id = mem.store([0.1, 0.2, ...], "label", "content")
        results = mem.retrieve([0.1, 0.2, ...], k=10)
        mem.consolidate()
        mem.shutdown()
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _find_lib()
        
        self._lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self._handle = None
    
    def _setup_functions(self):
        lib = self._lib
        
        # void* mazemaker_create_dim(int dim)
        lib.mazemaker_create_dim.argtypes = [ctypes.c_int]
        lib.mazemaker_create_dim.restype = ctypes.c_void_p
        
        # void* mazemaker_create(void)
        lib.mazemaker_create.argtypes = []
        lib.mazemaker_create.restype = ctypes.c_void_p
        
        # void mazemaker_destroy(void* handle)
        lib.mazemaker_destroy.argtypes = [ctypes.c_void_p]
        lib.mazemaker_destroy.restype = None
        
        # uint64_t mazemaker_store(void* handle, const float* vec, int dim,
        #                               const char* label, const char* content)
        lib.mazemaker_store.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.mazemaker_store.restype = ctypes.c_uint64
        
        # int mazemaker_retrieve_full(void* handle, const float* query, int dim,
        #                                  int k, MazemakerResult* results)
        lib.mazemaker_retrieve_full.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(CSearchResult)
        ]
        lib.mazemaker_retrieve_full.restype = ctypes.c_int
        
        # int mazemaker_search(void* handle, const char* query, int k,
        #                           MazemakerResult* results)
        lib.mazemaker_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(CSearchResult)
        ]
        lib.mazemaker_search.restype = ctypes.c_int
        
        # size_t mazemaker_consolidate(void* handle)
        lib.mazemaker_consolidate.argtypes = [ctypes.c_void_p]
        lib.mazemaker_consolidate.restype = ctypes.c_size_t
        
        # void mazemaker_stats_c(void* handle, MazemakerStats* stats)
        lib.mazemaker_stats_c.argtypes = [ctypes.c_void_p, ctypes.POINTER(CStats)]
        lib.mazemaker_stats_c.restype = None
        
        # int mazemaker_think_c(void* handle, uint64_t start_id, int depth,
        #                          uint64_t* ids, float* activations, int max_results)
        lib.mazemaker_think_c.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        lib.mazemaker_think_c.restype = ctypes.c_int

        # --- Graph Edge Operations (in-memory graph) ---

        # int mazemaker_add_edge(handle, from_id, to_id, weight, edge_type)
        lib.mazemaker_add_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_float, ctypes.c_char_p
        ]
        lib.mazemaker_add_edge.restype = ctypes.c_int

        # int mazemaker_batch_strengthen_edges(handle, from_ids, to_ids, count, delta)
        lib.mazemaker_batch_strengthen_edges.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_float
        ]
        lib.mazemaker_batch_strengthen_edges.restype = ctypes.c_int

        # int mazemaker_bulk_weaken_prune(handle, delta, threshold)
        lib.mazemaker_bulk_weaken_prune.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float
        ]
        lib.mazemaker_bulk_weaken_prune.restype = ctypes.c_int

        # int mazemaker_get_edges(handle, node_id, edge_ids, weights, max_edges)
        lib.mazemaker_get_edges.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        lib.mazemaker_get_edges.restype = ctypes.c_int

        # int64_t mazemaker_count_edges(handle)
        lib.mazemaker_count_edges.argtypes = [ctypes.c_void_p]
        lib.mazemaker_count_edges.restype = ctypes.c_int64
    
    def initialize(self, dim: int = 1024, hopfield_capacity: int = 1024,
                   episodic_capacity: int = 10000) -> bool:
        """Initialize the memory system."""
        self._handle = self._lib.mazemaker_create_dim(dim)
        return self._handle is not None
    
    def shutdown(self):
        """Shutdown and free resources."""
        if self._handle:
            self._lib.mazemaker_destroy(self._handle)
            self._handle = None
    
    def store(self, embedding: list[float], label: str = "", content: str = "") -> int:
        """Store a memory with embedding. Returns memory ID."""
        assert self._handle, "Not initialized. Call initialize() first."
        
        arr = (ctypes.c_float * len(embedding))(*embedding)
        label_bytes = label.encode('utf-8') if label else None
        content_bytes = content.encode('utf-8') if content else None
        
        return self._lib.mazemaker_store(
            self._handle, arr, len(embedding), label_bytes, content_bytes
        )
    
    def retrieve(self, query: list[float], k: int = 10) -> list[dict]:
        """Retrieve memories by embedding similarity."""
        assert self._handle, "Not initialized."
        
        arr = (ctypes.c_float * len(query))(*query)
        results = (CSearchResult * k)()
        
        count = self._lib.mazemaker_retrieve_full(
            self._handle, arr, len(query), k, results
        )
        
        return [
            {
                'id': results[i].id,
                'score': results[i].similarity,
                'similarity': results[i].similarity,
                'salience': results[i].salience,
                'label': results[i].label.decode('utf-8', errors='replace'),
                'content': results[i].content.decode('utf-8', errors='replace'),
            }
            for i in range(count)
        ]

    def search(self, query_text: str, k: int = 10) -> list[dict]:
        """Text-based search (uses internal embedding)."""
        assert self._handle, "Not initialized."
        
        results = (CSearchResult * k)()
        count = self._lib.mazemaker_search(
            self._handle, query_text.encode('utf-8'), k, results
        )
        
        return [
            {
                'id': results[i].id,
                'score': results[i].similarity,
                'similarity': results[i].similarity,
                'label': results[i].label.decode('utf-8', errors='replace'),
                'content': results[i].content.decode('utf-8', errors='replace'),
            }
            for i in range(count)
        ]

    def think(self, start_id: int, depth: int = 3, max_results: int = 20) -> list[dict]:
        """Spreading activation from a memory."""
        assert self._handle, "Not initialized."
        
        ids = (ctypes.c_uint64 * max_results)()
        activations = (ctypes.c_float * max_results)()
        
        count = self._lib.mazemaker_think_c(
            self._handle, start_id, depth, ids, activations, max_results
        )
        
        return [
            {
                'id': ids[i],
                'activation': activations[i],
                'label': f"node_{ids[i]}",
            }
            for i in range(count)
        ]
    
    def consolidate(self) -> int:
        """Run memory consolidation. Returns count of consolidated memories."""
        assert self._handle, "Not initialized."
        return self._lib.mazemaker_consolidate(self._handle)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        assert self._handle, "Not initialized."
        
        stats = CStats()
        self._lib.mazemaker_stats_c(self._handle, ctypes.byref(stats))
        
        return {
            'episodic_count': stats.episodic_count,
            'semantic_count': stats.semantic_count,
            'episodic_occupancy': stats.episodic_occupancy,
            'semantic_occupancy': stats.semantic_occupancy,
            'hopfield_patterns': stats.hopfield_patterns,
            'hopfield_occupancy': stats.hopfield_occupancy,
            'graph_nodes': stats.graph_nodes,
            'graph_edges': stats.graph_edges,
            'graph_density': stats.graph_density,
            'avg_store_us': stats.avg_store_us,
            'avg_retrieve_us': stats.avg_retrieve_us,
            'avg_search_us': stats.avg_search_us,
            'total_stores': stats.total_stores,
            'total_retrieves': stats.total_retrieves,
            'total_searches': stats.total_searches,
            'total_consolidations': stats.total_consolidations,
        }

    # --- Graph Edge Operations (in-memory graph) ---

    def add_edge(self, from_id: int, to_id: int, weight: float, edge_type: str = "similar") -> bool:
        """Add edge to the in-memory graph."""
        assert self._handle, "Not initialized."
        return bool(self._lib.mazemaker_add_edge(
            self._handle, from_id, to_id, weight, edge_type.encode('utf-8')
        ))

    def batch_strengthen_edges(self, edges: list[tuple[int, int]], delta: float = 0.05) -> int:
        """Batch strengthen edges by delta.

        Args:
            edges: list of (from_id, to_id) tuples
            delta: weight increment
        Returns:
            Number of edges updated
        """
        assert self._handle, "Not initialized."
        if not edges:
            return 0
        n = len(edges)
        from_arr = (ctypes.c_uint64 * n)(*([e[0] for e in edges]))
        to_arr = (ctypes.c_uint64 * n)(*([e[1] for e in edges]))
        return self._lib.mazemaker_batch_strengthen_edges(
            self._handle, from_arr, to_arr, n, delta
        )

    def bulk_weaken_prune(self, delta: float = 0.01, threshold: float = 0.05) -> int:
        """Bulk weaken all edges then prune below threshold."""
        assert self._handle, "Not initialized."
        return self._lib.mazemaker_bulk_weaken_prune(self._handle, delta, threshold)

    def get_edges(self, node_id: int, max_edges: int = 100) -> list[dict]:
        """Get all edges for a node."""
        assert self._handle, "Not initialized."
        edge_ids = (ctypes.c_uint64 * (max_edges * 2))()
        weights = (ctypes.c_float * max_edges)()

        count = self._lib.mazemaker_get_edges(
            self._handle, node_id, edge_ids, weights, max_edges
        )

        return [
            {
                'from_id': edge_ids[2 * i],
                'to_id': edge_ids[2 * i + 1],
                'weight': weights[i],
            }
            for i in range(count)
        ]

    def count_edges(self) -> int:
        """Count edges in the in-memory graph."""
        assert self._handle, "Not initialized."
        return self._lib.mazemaker_count_edges(self._handle)
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        self.shutdown()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    try:
        mem = MazemakerCpp()
        mem.initialize(dim=1024)
        print(f"Initialized C++ adapter")

        # Store some test data
        import random
        for i in range(5):
            vec = [random.uniform(-1, 1) for _ in range(1024)]
            norm = sum(x*x for x in vec) ** 0.5
            vec = [x/norm for x in vec]
            mid = mem.store(vec, f"test_{i}", f"Test memory {i}")
            print(f"  Stored: [{mid}] test_{i}")

        # Retrieve
        query = [random.uniform(-1, 1) for _ in range(1024)]
        norm = sum(x*x for x in query) ** 0.5
        query = [x/norm for x in query]
        results = mem.retrieve(query, k=3)
        print(f"\n  Retrieved {len(results)} results")
        for r in results:
            print(f"    [{r['id']}] {r['label']}: {r['score']:.3f}")
        
        # Stats
        stats = mem.get_stats()
        print(f"\n  Stats: {stats}")
        
        mem.shutdown()
        print("\nC++ bridge: OK")
    except FileNotFoundError as e:
        print(f"C++ library not found: {e}")
        print("Build first: cd ~/projects/mazemaker-adapter/build && cmake --build . -j$(nproc)")
