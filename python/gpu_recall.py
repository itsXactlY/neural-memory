"""GPU-accelerated vector recall for neural memory.

Loads pre-computed embeddings onto GPU for sub-millisecond cosine similarity search.
Much faster than Python loop or C++ Hopfield network.

Usage:
    engine = GpuRecallEngine()
    engine.load()
    results = engine.recall("query text", k=5)
"""

import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE_DIR = Path.home() / ".neural_memory" / "gpu_cache"
_EMBEDDINGS_PATH = _CACHE_DIR / "embeddings.npy"
_METADATA_PATH = _CACHE_DIR / "metadata.pkl"


class GpuRecallEngine:
    """GPU-accelerated cosine similarity search over neural memory embeddings."""

    def __init__(self):
        self._device = None
        self._emb_tensor = None  # (N, dim) float32 on GPU
        # Cached row-normalised view of _emb_tensor. Built lazily the first
        # time recall() encounters a non-unit query (so we don't pay the
        # normalisation cost on already-normalised models like e5-large).
        self._emb_tensor_normed = None
        self._ids = []
        self._labels = []
        self._contents = []
        self._dim = 1024
        self._loaded = False

    def load(self, embed_fn=None) -> bool:
        """Load embeddings onto GPU.

        Args:
            embed_fn: Optional callable(text) -> list[float] for query embedding.
                      If None, uses sentence-transformers.

        Returns:
            True if loaded successfully.
        """
        import torch

        # Select device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Load cached embeddings
        if not _EMBEDDINGS_PATH.exists():
            return False

        emb_array = np.load(str(_EMBEDDINGS_PATH))
        self._dim = emb_array.shape[1]

        with open(str(_METADATA_PATH), "rb") as f:
            meta = pickle.load(f)

        self._ids = meta["ids"]
        self._labels = meta["labels"]
        self._contents = meta["contents"]

        # Move to GPU
        self._emb_tensor = torch.tensor(emb_array, device=self._device, dtype=torch.float32)
        # Invalidate any prior normalised cache — a reload (e.g. after the
        # source DB grew) means the row layout changed.
        self._emb_tensor_normed = None

        # Store embed function
        self._embed_fn = embed_fn

        self._loaded = True
        return True

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Search memories by semantic similarity.

        Args:
            query: Search query text.
            k: Number of results.

        Returns:
            List of {id, label, content, similarity} dicts.
        """
        if not self._loaded:
            return []

        import torch

        # Embed query
        if self._embed_fn is None:
            raise RuntimeError("No embed function configured")

        query_vec = self._embed_fn(query)
        # Dim guard: a query produced by the active backend (e.g. 1024-d
        # FastEmbed) against a GPU cache that was populated by a different
        # backend (e.g. 384-d MiniLM) would crash inside torch.matmul. The
        # outer try/except in memory_client catches the crash, but doing
        # the check up-front skips the exception machinery and yields a
        # clean fast path through the CPU/HNSW fallbacks instead.
        if len(query_vec) != self._dim:
            return []
        q = torch.tensor(query_vec, device=self._device, dtype=torch.float32)

        # Normalize if needed (check magnitude). For models that already
        # emit unit-norm vectors (FastEmbed e5-large, sentence-transformers
        # with normalize_embeddings=True), this branch is skipped and we
        # use the cached tensor directly.
        mag = torch.norm(q)
        if mag > 2.0:  # Raw embedding, needs normalization
            q = q / mag
            # Cache the row-normalised tensor across calls so subsequent
            # raw-embedding queries don't re-norm the entire (N, dim) matrix
            # on every recall. Built once lazily; survives until the engine
            # reloads (which clears it via clear_cache()).
            if self._emb_tensor_normed is None:
                norms = torch.norm(self._emb_tensor, dim=1, keepdim=True)
                # Avoid divide-by-zero on any zero-row pathology.
                norms = torch.clamp(norms, min=1e-12)
                self._emb_tensor_normed = self._emb_tensor / norms
            emb_normed = self._emb_tensor_normed
        else:
            emb_normed = self._emb_tensor

        # Cosine similarity (dot product of normalized vectors)
        sims = torch.matmul(emb_normed, q)

        # Top-k
        top_k = torch.topk(sims, min(k, len(self._ids)))

        results = []
        for idx, sim in zip(top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()):
            results.append({
                "id": self._ids[idx],
                "label": self._labels[idx],
                "content": self._contents[idx],
                "similarity": float(sim),
            })

        return results

    def stats(self) -> dict:
        """Return engine stats."""
        return {
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
            "memories": len(self._ids),
            "dim": self._dim,
            "vram_mb": self._emb_tensor.element_size() * self._emb_tensor.nelement() / 1024 / 1024 if self._emb_tensor is not None else 0,
        }

    def shutdown(self):
        """Free GPU memory."""
        if self._emb_tensor is not None:
            del self._emb_tensor
            self._emb_tensor = None
        if self._emb_tensor_normed is not None:
            del self._emb_tensor_normed
            self._emb_tensor_normed = None
        if self._device and self._device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        self._loaded = False
