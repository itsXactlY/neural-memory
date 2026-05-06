"""GPU-accelerated vector recall for mazemaker.

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

_CACHE_DIR = Path.home() / ".mazemaker" / "engine" / "gpu_cache"
_EMBEDDINGS_PATH = _CACHE_DIR / "embeddings.npy"
_METADATA_PATH = _CACHE_DIR / "metadata.pkl"


class GpuRecallEngine:
    """GPU-accelerated cosine similarity search over mazemaker embeddings."""

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

    def add_one(self, mem_id: int, label: str, content: str, embedding) -> None:
        """Append one new memory to the in-GPU tensor. Called from
        Mazemaker.remember() so newly-stored memories are searchable on
        GPU immediately (no cache rebuild needed).

        Embedding can be list[float], np.ndarray, or torch.Tensor.
        """
        if not self._loaded:
            return
        import torch
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, device=self._device, dtype=torch.float32)
        else:
            embedding = embedding.to(self._device, dtype=torch.float32)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        self._emb_tensor = torch.cat([self._emb_tensor, embedding], dim=0)
        # Invalidate normalised cache; recall() will recompute on next miss.
        self._emb_tensor_normed = None
        self._ids.append(int(mem_id))
        self._labels.append(label or "")
        self._contents.append(content or "")

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

        # Always row-normalise the stored tensor + the query before the
        # matmul. The DB can carry mixed-magnitude rows (some written by
        # an older engine that didn't pass normalize_embeddings=True; some
        # from bulk imports that did). Without normalising both sides we
        # get raw projection scores in arbitrary range — REM's filter of
        # `0.3 < sim < 0.95` then drops every candidate. Cost is one-time
        # per process: norms cached in self._emb_tensor_normed.
        if self._emb_tensor_normed is None:
            norms = torch.norm(self._emb_tensor, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            self._emb_tensor_normed = self._emb_tensor / norms
        emb_normed = self._emb_tensor_normed
        q_mag = torch.norm(q)
        if q_mag > 1e-12:
            q = q / q_mag

        # Cosine similarity (dot product of normalized vectors) — always
        # in [-1, 1] now, so REM's similarity-window filter behaves.
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
