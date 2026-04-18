#!/usr/bin/env python3
"""
lstm_knn_bridge.py - Python interface to C++ LSTM+kNN via ctypes.

Provides:
- LSTMBridge: Wraps C++ LSTMPredictor for next-embedding prediction
- KNNBridge: Wraps C++ KNNEngine for multi-signal scoring
- enhanced_recall: Integration function combining LSTM + kNN + AccessLogger

Follows same ctypes pattern as cpp_bridge.py.
"""

import ctypes
import ctypes.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ============================================================================
# Find library (same pattern as cpp_bridge.py)
# ============================================================================

def _find_lib() -> str:
    candidates = [
        Path(__file__).parent.parent / "build" / "libneural_memory.so",
        Path.home() / "projects" / "neural-memory-adapter" / "build" / "libneural_memory.so",
        Path("/usr/local/lib/libneural_memory.so"),
        Path("/usr/lib/libneural_memory.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    lib = ctypes.util.find_library("neural_memory")
    if lib:
        return lib
    raise FileNotFoundError(
        "libneural_memory.so not found. Build first:\n"
        "  cd ~/projects/neural-memory-adapter/build && cmake --build . -j$(nproc)"
    )


# ============================================================================
# C API types
# ============================================================================

class CKNNResult(ctypes.Structure):
    """Must match KNNCResult in c_api.h exactly."""
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
        ("embed_similarity", ctypes.c_float),
        ("temporal_score", ctypes.c_float),
        ("freq_score", ctypes.c_float),
        ("graph_score", ctypes.c_float),
    ]


# ============================================================================
# ScoredResult dataclass for Python API
# ============================================================================

@dataclass
class ScoredResult:
    """Python-friendly search result."""
    id: int
    score: float
    embed_similarity: float
    temporal_score: float
    freq_score: float
    graph_score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "embed_similarity": self.embed_similarity,
            "temporal_score": self.temporal_score,
            "freq_score": self.freq_score,
            "graph_score": self.graph_score,
        }


# ============================================================================
# LSTMBridge
# ============================================================================

class LSTMPredictor:
    """Wraps C++ LSTMPredictor via ctypes.

    Usage:
        lstm = LSTMPredictor(input_dim=1024, hidden_dim=256)
        prediction = lstm.predict_next(sequence_embeddings)
        loss = lstm.train_on_pair(sequence_embeddings, target_embedding)
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _find_lib()

        self._lib = ctypes.CDLL(lib_path)
        self._handle = None
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._setup_functions()

        # Create the LSTM
        self._handle = self._lib.nm_lstm_create(input_dim, hidden_dim)
        if not self._handle:
            raise RuntimeError("Failed to create LSTM predictor")

    def _setup_functions(self):
        lib = self._lib

        # void* nm_lstm_create(int input_dim, int hidden_dim)
        lib.nm_lstm_create.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.nm_lstm_create.restype = ctypes.c_void_p

        # int nm_lstm_forward(void* handle, const float* sequence, int seq_len, float* output)
        lib.nm_lstm_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.nm_lstm_forward.restype = ctypes.c_int

        # float nm_lstm_train(void* handle, const float* sequence, int seq_len,
        #                      const float* target, float lr)
        lib.nm_lstm_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_float,
        ]
        lib.nm_lstm_train.restype = ctypes.c_float

        # int nm_lstm_save(void* handle, const char* path)
        lib.nm_lstm_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nm_lstm_save.restype = ctypes.c_int

        # void* nm_lstm_load(const char* path, int input_dim, int hidden_dim)
        lib.nm_lstm_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.nm_lstm_load.restype = ctypes.c_void_p

        # void nm_lstm_destroy(void* handle)
        lib.nm_lstm_destroy.argtypes = [ctypes.c_void_p]
        lib.nm_lstm_destroy.restype = None

    def predict_next(self, sequence: list[list[float]]) -> list[float]:
        """Predict next embedding from a sequence of embeddings.

        Args:
            sequence: List of embedding vectors, each length input_dim.
        Returns:
            Predicted next embedding (length input_dim).
        """
        assert self._handle, "LSTM not initialized"
        seq_len = len(sequence)
        if seq_len == 0:
            raise ValueError("Empty sequence")

        # Validate embedding dimensions to prevent out-of-bounds C reads
        for i, emb in enumerate(sequence):
            if len(emb) != self._input_dim:
                raise ValueError(
                    f"Embedding at index {i} has dim {len(emb)}, expected {self._input_dim}"
                )

        # Flatten sequence to C array
        flat = [v for emb in sequence for v in emb]
        seq_arr = (ctypes.c_float * len(flat))(*flat)
        out_arr = (ctypes.c_float * self._input_dim)()

        rc = self._lib.nm_lstm_forward(
            self._handle, seq_arr, seq_len, out_arr
        )
        if rc != 0:
            raise RuntimeError("LSTM forward pass failed")

        return list(out_arr)

    def train_on_pair(self, sequence: list[list[float]],
                      target: list[float], lr: float = 0.001) -> float:
        """Train LSTM on one (sequence, target) pair.

        Args:
            sequence: Input sequence of embeddings.
            target: Expected next embedding.
            lr: Learning rate.
        Returns:
            MSE loss value.
        """
        assert self._handle, "LSTM not initialized"
        seq_len = len(sequence)
        if seq_len == 0:
            raise ValueError("Empty sequence")

        # Validate dimensions to prevent out-of-bounds C reads
        for i, emb in enumerate(sequence):
            if len(emb) != self._input_dim:
                raise ValueError(
                    f"Embedding at index {i} has dim {len(emb)}, expected {self._input_dim}"
                )
        if len(target) != self._input_dim:
            raise ValueError(
                f"Target has dim {len(target)}, expected {self._input_dim}"
            )

        flat = [v for emb in sequence for v in emb]
        seq_arr = (ctypes.c_float * len(flat))(*flat)
        tgt_arr = (ctypes.c_float * self._input_dim)(*target)

        loss = self._lib.nm_lstm_train(
            self._handle, seq_arr, seq_len, tgt_arr, lr
        )
        if loss < 0:
            raise RuntimeError("LSTM training failed")
        return loss

    def save(self, path: str):
        """Save LSTM weights to file."""
        assert self._handle, "LSTM not initialized"
        rc = self._lib.nm_lstm_save(self._handle, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to save LSTM to {path}")

    def load(self, path: str):
        """Load LSTM weights from file."""
        assert self._handle, "LSTM not initialized"
        # Save old handle — destroy only AFTER successful load to preserve object validity
        old_handle = self._handle
        new_handle = self._lib.nm_lstm_load(
            path.encode("utf-8"), self._input_dim, self._hidden_dim
        )
        if not new_handle:
            # Load failed — keep old handle alive, don't leave object broken
            raise RuntimeError(f"Failed to load LSTM from {path}")
        # Success — destroy old handle, install new one
        self._lib.nm_lstm_destroy(old_handle)
        self._handle = new_handle

    def close(self):
        """Destroy the LSTM predictor."""
        if getattr(self, '_handle', None):
            self._lib.nm_lstm_destroy(self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, '_handle', None):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Alias for backward compatibility
LSTMBridge = LSTMPredictor


# ============================================================================
# KNNBridge
# ============================================================================

class KNNEngine:
    """Wraps C++ KNNEngine via ctypes.

    Usage:
        knn = KNNEngine(embed_dim=1024)
        results = knn.search(query, candidates, k=10)
    """

    def __init__(self, embed_dim: int = 1024, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _find_lib()

        self._lib = ctypes.CDLL(lib_path)
        self._handle = None
        self._embed_dim = embed_dim
        self._setup_functions()

        # Create the kNN engine
        self._handle = self._lib.nm_knn_create(embed_dim)
        if not self._handle:
            raise RuntimeError("Failed to create kNN engine")

    def _setup_functions(self):
        lib = self._lib

        # void* nm_knn_create(int embed_dim)
        lib.nm_knn_create.argtypes = [ctypes.c_int]
        lib.nm_knn_create.restype = ctypes.c_void_p

        # int nm_knn_search(handle, query, candidates, candidate_ids,
        #                    count, k, timestamps, access_counts, graph_scores,
        #                    lstm_context, results)
        lib.nm_knn_search.argtypes = [
            ctypes.c_void_p,                          # handle
            ctypes.POINTER(ctypes.c_float),           # query
            ctypes.c_int,                              # embed_dim
            ctypes.POINTER(ctypes.c_float),           # candidates
            ctypes.POINTER(ctypes.c_uint64),          # candidate_ids
            ctypes.c_int,                              # count
            ctypes.c_int,                              # k
            ctypes.POINTER(ctypes.c_float),           # timestamps
            ctypes.POINTER(ctypes.c_float),           # access_counts
            ctypes.POINTER(ctypes.c_float),           # graph_scores
            ctypes.POINTER(ctypes.c_float),           # lstm_context (nullable)
            ctypes.POINTER(CKNNResult),               # results
        ]
        lib.nm_knn_search.restype = ctypes.c_int

        # void nm_knn_adjust_weights(handle, lstm_context)
        lib.nm_knn_adjust_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.nm_knn_adjust_weights.restype = None

        # void nm_knn_destroy(handle)
        lib.nm_knn_destroy.argtypes = [ctypes.c_void_p]
        lib.nm_knn_destroy.restype = None

    def search(self, query: list[float],
               candidates: list[list[float]],
               candidate_ids: list[int],
               k: int = 10,
               timestamps: Optional[list[float]] = None,
               access_counts: Optional[list[float]] = None,
               graph_scores: Optional[list[float]] = None,
               lstm_context: Optional[list[float]] = None) -> list[ScoredResult]:
        """Search candidates with multi-signal scoring.

        Args:
            query: Query embedding vector.
            candidates: List of candidate embedding vectors.
            candidate_ids: Memory ID for each candidate.
            k: Number of results to return.
            timestamps: Creation/access time for each candidate (epoch seconds).
            access_counts: Access frequency for each candidate.
            graph_scores: Graph relevance score for each candidate (0-1).
            lstm_context: Optional LSTM-predicted embedding for re-ranking.
        Returns:
            List of ScoredResult sorted by combined score descending.
        """
        assert self._handle, "kNN engine not initialized"
        count = len(candidates)
        if count == 0:
            return []
        if len(candidate_ids) != count:
            raise ValueError("candidate_ids length must match candidates")
        if len(query) != self._embed_dim:
            raise ValueError(f"query has dim {len(query)}, expected {self._embed_dim}")
        # Clamp k to available candidates to prevent undefined C behavior
        k = min(k, count)

        # Default arrays
        now = time.time()
        if timestamps is None:
            timestamps = [now] * count
        if access_counts is None:
            access_counts = [1.0] * count
        if graph_scores is None:
            graph_scores = [0.0] * count

        # Build C arrays
        query_arr = (ctypes.c_float * self._embed_dim)(*query)
        flat_cands = [v for c in candidates for v in c]
        cands_arr = (ctypes.c_float * len(flat_cands))(*flat_cands)
        ids_arr = (ctypes.c_uint64 * count)(*candidate_ids)
        ts_arr = (ctypes.c_float * count)(*timestamps)
        acc_arr = (ctypes.c_float * count)(*access_counts)
        graph_arr = (ctypes.c_float * count)(*graph_scores)
        results_arr = (CKNNResult * k)()

        # LSTM context pointer
        ctx_ptr = None
        if lstm_context is not None:
            ctx_ptr = (ctypes.c_float * self._embed_dim)(*lstm_context)

        n = self._lib.nm_knn_search(
            self._handle, query_arr, self._embed_dim,
            cands_arr, ids_arr, count, k,
            ts_arr, acc_arr, graph_arr,
            ctx_ptr, results_arr
        )
        if n < 0:
            raise RuntimeError("kNN search failed")

        return [
            ScoredResult(
                id=results_arr[i].id,
                score=results_arr[i].score,
                embed_similarity=results_arr[i].embed_similarity,
                temporal_score=results_arr[i].temporal_score,
                freq_score=results_arr[i].freq_score,
                graph_score=results_arr[i].graph_score,
            )
            for i in range(n)
        ]

    def adjust_weights(self, lstm_context: Optional[list[float]] = None):
        """Adjust scoring weights based on LSTM context availability.

        Args:
            lstm_context: LSTM embedding if available (boosts temporal weight),
                         None to reset to defaults.
        """
        assert self._handle, "kNN engine not initialized"
        ctx_ptr = None
        if lstm_context is not None:
            ctx_ptr = (ctypes.c_float * self._embed_dim)(*lstm_context)
        self._lib.nm_knn_adjust_weights(self._handle, ctx_ptr)

    def close(self):
        """Destroy the kNN engine."""
        if self._handle:
            self._lib.nm_knn_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Alias for backward compatibility
KNNBridge = KNNEngine


# ============================================================================
# Integration function
# ============================================================================

def enhanced_recall(query_embedding: list[float],
                    memories: list[dict],
                    access_logger,
                    lstm_bridge: LSTMPredictor,
                    knn_bridge: KNNEngine,
                    k: int = 10) -> list[ScoredResult]:
    """Enhanced recall combining LSTM prediction with multi-signal kNN.

    Flow:
    1. Log the access event to access_logger
    2. Get LSTM prediction from recent access sequence
    3. Pass LSTM context to kNN engine for re-ranking
    4. Return re-ranked results

    Args:
        query_embedding: Current query vector (e.g., 1024-dim).
        memories: List of candidate memory dicts with keys:
                  'embedding', 'id', 'timestamp', 'access_count', 'graph_score'.
        access_logger: AccessLogger instance.
        lstm_bridge: LSTMPredictor for temporal prediction.
        knn_bridge: KNNEngine for multi-signal scoring.
        k: Number of results to return.
    Returns:
        List of ScoredResult sorted by combined score.
    """
    # Log the access
    # (Don't log yet — we'll log once after search with actual result IDs/scores)

    # Get LSTM prediction from recent sequence
    lstm_context = None
    recent = access_logger.get_sequence(n=20)
    if len(recent) >= 2:
        try:
            seq_embeddings = [e["query_emb"] for e in recent[-10:]]
            lstm_context = lstm_bridge.predict_next(seq_embeddings)
        except Exception as e:
            print(f"[enhanced_recall] LSTM prediction failed: {e}")

    # Prepare candidate data
    if not memories:
        return []

    candidates = [m["embedding"] for m in memories]
    candidate_ids = [m["id"] for m in memories]
    timestamps = [m.get("timestamp", time.time()) for m in memories]
    access_counts = [float(m.get("access_count", 1)) for m in memories]
    graph_scores = [m.get("graph_score", 0.0) for m in memories]

    # kNN search with LSTM context
    results = knn_bridge.search(
        query=query_embedding,
        candidates=candidates,
        candidate_ids=candidate_ids,
        k=k,
        timestamps=timestamps,
        access_counts=access_counts,
        graph_scores=graph_scores,
        lstm_context=lstm_context,
    )

    # Update access log with actual results
    if results:
        access_logger.log_recall(
            query_embedding=query_embedding,
            result_ids=[r.id for r in results],
            result_scores=[r.score for r in results],
        )

    return results


if __name__ == "__main__":
    print("lstm_knn_bridge.py - Python bindings for LSTM+kNN C++ engine")
    print("Import this module to use LSTMPredictor, KNNEngine, and enhanced_recall")
