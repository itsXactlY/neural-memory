#!/usr/bin/env python3
"""
embed_provider.py - Text Embedding for Neural Memory Adapter

PRIMARY: FastEmbed + ONNX Runtime (no PyTorch, no CUDA conflict)
FALLBACK: sentence-transformers (if FastEmbed unavailable)

SHARED MODE (default): First process starts a UNIX socket server holding
the model. All other processes connect as clients. ONE model instance
for ALL hermes sessions.

Env vars:
  EMBED_MODEL        — model name (default: intfloat/multilingual-e5-large)
  EMBED_BACKEND      — force backend: fastembed|sentence-transformers|tfidf|hash
  EMBED_SOCKET       — UNIX socket path (default: ~/.neural_memory/embed.sock)
  EMBED_NO_SHARED    — set to disable shared server mode
"""

import os
import sys
import pickle
import hashlib
import re
import json
import socket
import struct
import time
import threading
from pathlib import Path

CACHE_DIR = Path.home() / ".neural_memory"
CACHE_FILE = CACHE_DIR / "embed_cache.pkl"
MODEL_DIR = CACHE_DIR / "models"
SOCKET_PATH = Path(os.environ.get('EMBED_SOCKET', str(CACHE_DIR / "embed.sock")))
DIMENSION = 1024  # Default dim for e5-large / bge-large


'''
# aLca TODO :: figure out what polars can improve here
as polars will bring another dependency (parallel multithreading but...) on top what would basically only here over pure py
so a question is how can an another layer of cache help here on multiagent multitasking level
when we hit it with an podcast from jackrabbits wonderland, or other heavy contents, and can one gateway handle it while 
caged knee deep inside its own virtual machine, getting in parallel spoonfed heavy chunks ...

another idea maybe outsource further the embedding to a microservice, which can be scaled independently, and can use more powerful hardware if needed
this way, the main application can remain lightweight and responsive, while the embedding service can handle the heavy lifting. 
we can use the wonderland for the embedding service, and then the main application can make HTTP requests to get embeddings as needed. 
this also allows for better fault tolerance ontop as the embedding service can be restarted or scaled without affecting the main application - hello my dearest nightmare rn
'''


# ============================================================================
# FastEmbed Backend (ONNX Runtime — NO PyTorch)
# ============================================================================

class FastEmbedBackend:
    """Uses fastembed (ONNX Runtime) — no PyTorch, no CUDA conflict.
    
    Default model: intfloat/multilingual-e5-large (1024d, multilingual)
    ~50ms per embedding, ~2s model load, ~200MB disk.
    """
    MODEL_NAME = os.environ.get('EMBED_MODEL', 'intfloat/multilingual-e5-large')
    
    def __init__(self, model_name=None):
        from fastembed import TextEmbedding
        from pathlib import Path as P
        
        self._model_name = model_name or self.MODEL_NAME
        cache_dir = str(MODEL_DIR)
        
        print(f"[fastembed] Loading {self._model_name}...")
        t0 = time.time()
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=cache_dir,
        )
        # Probe for dim
        probe = list(self._model.embed(["probe"]))
        self._dim = len(probe[0])
        print(f"[fastembed] Ready ({self._dim}d, {time.time()-t0:.1f}s)")
    
    @property
    def dim(self):
        return self._dim
    
    def embed(self, text: str) -> list[float]:
        vecs = list(self._model.embed([text]))
        return vecs[0].tolist() if hasattr(vecs[0], 'tolist') else list(vecs[0])
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = list(self._model.embed(texts))
        return [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vecs]


# ============================================================================
# Shared Embed Server (UNIX socket)
# ============================================================================

class SharedEmbedServer:
    """UNIX socket server that holds the embedding model.
    
    Protocol: length-prefixed JSON over UNIX socket.
    Request:  {"cmd": "embed", "text": "..."} or {"cmd": "embed_batch", "texts": [...]}
    Response: {"ok": true, "vec": [...]} or {"ok": true, "vecs": [[...], ...]}
    Error:    {"ok": false, "error": "..."}
    """
    
    def __init__(self, model_name=None, idle_timeout=20):
        self.model_name = model_name or os.environ.get('EMBED_MODEL', 'intfloat/multilingual-e5-large')
        self.idle_timeout = idle_timeout
        self._backend = None
        self.dim = None
        self._last_used = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._sock = None
    
    def start(self):
        """Load model and start listening. Returns True if started, False if already running."""
        if SOCKET_PATH.exists():
            try:
                client = SharedEmbedClient()
                if client.ping():
                    print(f"[embed-server] Already running at {SOCKET_PATH}")
                    return False
            except:
                pass
            SOCKET_PATH.unlink()
        
        self._load_model()
        self._start_listener()
        self._start_eject_timer()
        print(f"[embed-server] Listening at {SOCKET_PATH}")
        return True
    
    def _load_model(self):
        self._backend = FastEmbedBackend(model_name=self.model_name)
        self.dim = self._backend.dim
        self._last_used = time.time()
    
    def _start_listener(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(SOCKET_PATH))
        self._sock.listen(8)
        self._sock.settimeout(1.0)
        self._running = True
        
        t = threading.Thread(target=self._accept_loop, daemon=True, name="embed-server")
        t.start()
    
    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    continue
                break
    
    def _handle(self, conn):
        try:
            while True:
                raw_len = conn.recv(4)
                if not raw_len:
                    break
                msg_len = struct.unpack('!I', raw_len)[0]
                data = b''
                while len(data) < msg_len:
                    chunk = conn.recv(min(msg_len - len(data), 65536))
                    if not chunk:
                        break
                    data += chunk
                
                req = json.loads(data)
                resp = self._process(req)
                
                resp_bytes = json.dumps(resp).encode()
                conn.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)
        except Exception:
            pass
        finally:
            conn.close()
    
    def _process(self, req):
        cmd = req.get("cmd")
        self._last_used = time.time()
        
        with self._lock:
            try:
                if cmd == "embed":
                    vec = self._backend.embed(req["text"])
                    return {"ok": True, "vec": vec}
                elif cmd == "embed_batch":
                    vecs = self._backend.embed_batch(req["texts"])
                    return {"ok": True, "vecs": vecs}
                elif cmd == "status":
                    return {
                        "ok": True, "model": self.model_name, "dim": self.dim,
                        "idle": round(time.time() - self._last_used, 1),
                        "timeout": self.idle_timeout,
                        "backend": "fastembed",
                    }
                elif cmd == "ping":
                    return {"ok": True, "dim": self.dim}
                else:
                    return {"ok": False, "error": f"unknown cmd: {cmd}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}
    
    def _start_eject_timer(self):
        # No-op for FastEmbed (CPU only, nothing to eject)
        pass
    
    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()


# ============================================================================
# Shared Embed Client
# ============================================================================

class SharedEmbedClient:
    """Client that connects to SharedEmbedServer via UNIX socket."""
    
    def __init__(self, timeout=10.0):
        self._sock = None
        self._dim = None
        self._timeout = timeout
        self._connect()
    
    def _connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(str(SOCKET_PATH))
        resp = self._send({"cmd": "ping"})
        self._dim = resp.get("dim", 1024)
    
    def _send(self, req):
        msg = json.dumps(req).encode()
        self._sock.sendall(struct.pack('!I', len(msg)) + msg)
        raw_len = self._sock.recv(4)
        resp_len = struct.unpack('!I', raw_len)[0]
        data = b''
        while len(data) < resp_len:
            chunk = self._sock.recv(min(resp_len - len(data), 65536))
            if not chunk:
                break
            data += chunk
        resp = json.loads(data)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown error"))
        return resp
    
    @property
    def dim(self):
        return self._dim
    
    def embed(self, text):
        resp = self._send({"cmd": "embed", "text": text})
        return resp["vec"]
    
    def embed_batch(self, texts):
        resp = self._send({"cmd": "embed_batch", "texts": texts})
        return resp["vecs"]
    
    def ping(self):
        try:
            self._send({"cmd": "ping"})
            return True
        except:
            return False
    
    def close(self):
        if self._sock:
            self._sock.close()


# ============================================================================
# SentenceTransformerBackend (fallback — requires PyTorch)
# ============================================================================

class SentenceTransformerBackend:
    """Fallback: uses sentence-transformers (requires PyTorch).
    
    Only loaded if FastEmbed is unavailable.
    """
    MODEL_NAME = os.environ.get('EMBED_MODEL', 'intfloat/multilingual-e5-large')
    FORCED_DEVICE = os.environ.get('EMBED_DEVICE', None)
    
    _shared_model = None
    _shared_dim = None
    _lock = None
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        
        if SentenceTransformerBackend._lock is None:
            SentenceTransformerBackend._lock = threading.Lock()
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        device = 'cpu'
        if self.FORCED_DEVICE:
            device = self.FORCED_DEVICE
        elif torch.cuda.is_available():
            try:
                free = torch.cuda.mem_get_info(0)[0] / 1024**2
                if free > 500:
                    device = 'cuda'
            except:
                pass
        
        safe_name = self.MODEL_NAME.replace('/', '--')
        cache_base = MODEL_DIR / f"models--{safe_name}"
        model_path = None
        refs_main = cache_base / "refs" / "main"
        if refs_main.exists():
            snapshot_hash = refs_main.read_text().strip()
            snap = cache_base / "snapshots" / snapshot_hash
            if (snap / "config.json").exists():
                model_path = str(snap)
        if model_path is None:
            snapshots_dir = cache_base / "snapshots"
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    if (snap / "config.json").exists():
                        model_path = str(snap)
                        break
        if model_path is None:
            raise FileNotFoundError(f"No cached model: {self.MODEL_NAME}")
        
        print(f"[embed] Loading {model_path} directly on {device}...")
        self.model = SentenceTransformer(model_path, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        SentenceTransformerBackend._shared_model = self.model
        SentenceTransformerBackend._shared_dim = self.dim
        print(f"[embed] {self.MODEL_NAME} ready ({self.dim}d)")
    
    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ============================================================================
# TF-IDF+SVD Backend (pure numpy fallback)
# ============================================================================

class TfidfSvdBackend:
    """Pure numpy TF-IDF + SVD embedding (no ML dependencies)"""
    def __init__(self, dim: int = DIMENSION):
        import numpy as np
        self.np = np
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None
        self._trained = False
        self._corpus: list[str] = []
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def _hash_embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            h = hash(token)
            for j in range(4):
                idx = (h ^ (j * 2654435761)) % self.dim
                vec[idx] += 1.0 / (1.0 + i * 0.1)
                h = (h >> 8) | ((h & 0xFF) << 24)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    
    def fit(self, texts: list[str]):
        np = self.np
        doc_freq = {}
        all_tokens = []
        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.append(list(tokens))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        
        sorted_vocab = sorted(doc_freq.items(), key=lambda x: -x[1])[:10000]
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_vocab)}
        vocab_size = len(self.vocab)
        
        n_docs = len(texts)
        self.idf = np.zeros(vocab_size)
        for word, idx in self.vocab.items():
            self.idf[idx] = np.log((n_docs + 1) / (doc_freq.get(word, 1) + 1)) + 1
        
        tfidf = np.zeros((n_docs, vocab_size))
        for i, tokens in enumerate(all_tokens):
            for t in tokens:
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] += 1
            tfidf[i] *= self.idf
            norm = np.linalg.norm(tfidf[i])
            if norm > 0:
                tfidf[i] /= norm
        
        if vocab_size > self.dim:
            U, S, Vt = np.linalg.svd(tfidf[:, :min(vocab_size, 5000)], full_matrices=False)
            self.svd_components = Vt[:self.dim].T
        else:
            self.svd_components = np.eye(vocab_size, self.dim)
        
        self._trained = True
    
    def embed(self, text: str) -> list[float]:
        self._corpus.append(text)
        if not self._trained:
            if len(self._corpus) >= 5:
                self.fit(self._corpus)
            return self._hash_embed(text)
        
        np = self.np
        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                vec[self.vocab[t]] += 1
        if self.idf is not None:
            vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        if self.svd_components is not None:
            vec = vec @ self.svd_components
        result = np.zeros(self.dim)
        result[:min(len(vec), self.dim)] = vec[:self.dim]
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================================
# Hash Backend (zero dependencies)
# ============================================================================

class HashBackend:
    """Simple hash-based embedding (zero dependencies)"""
    def __init__(self, dim: int = DIMENSION):
        self.dim = dim
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            h = hash(token)
            for j in range(3):
                idx = (h ^ (j * 2654435761)) % self.dim
                vec[idx] += 1.0 / (1.0 + i * 0.1)
                h = (h >> 8) | ((h & 0xFF) << 24)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================================
# Provider with caching
# ============================================================================

class EmbeddingProvider:
    def __init__(self, backend: str = "auto"):
        self.cache: dict[str, list[float]] = {}
        self._load_cache()
        
        if backend == "auto":
            self.backend = self._auto_detect()
        elif backend == "fastembed":
            self.backend = FastEmbedBackend()
        elif backend == "sentence-transformers":
            self.backend = SentenceTransformerBackend()
        elif backend == "tfidf":
            self.backend = TfidfSvdBackend()
        else:
            self.backend = HashBackend()
        
        print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d)")
    
    def _auto_detect(self):
        """Auto-detect best available backend.
        Priority: FastEmbed (ONNX) > sentence-transformers (PyTorch) > TF-IDF > hash
        """
        # Check for forced backend
        forced = os.environ.get('EMBED_BACKEND', '').lower()
        if forced == 'fastembed':
            return FastEmbedBackend()
        elif forced == 'sentence-transformers':
            return SentenceTransformerBackend()
        elif forced == 'tfidf':
            return TfidfSvdBackend()
        elif forced == 'hash':
            return HashBackend()
        
        # Try FastEmbed first (ONNX — no PyTorch, no CUDA conflict)
        try:
            import fastembed
            backend = FastEmbedBackend()
            print(f"[embed] Auto-selected: fastembed (ONNX)")
            return backend
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"[embed] fastembed failed: {e}", file=sys.stderr)
        
        # Fallback: sentence-transformers (requires PyTorch)
        try:
            import sentence_transformers
            import torch
            backend = SentenceTransformerBackend()
            print(f"[embed] Auto-selected: sentence-transformers (PyTorch)")
            return backend
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"[embed] sentence-transformers failed: {e}", file=sys.stderr)
        
        # TF-IDF+SVD
        try:
            import numpy
            print("[embed] Auto-selected: TF-IDF+SVD")
            return TfidfSvdBackend()
        except ImportError:
            pass
        
        # Hash fallback
        print("[embed] Auto-selected: hash (zero dependencies)")
        return HashBackend()
    
    def _load_cache(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}
    
    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        
        vec = self.backend.embed(text)
        self.cache[key] = vec

        # Save periodically
        if len(self.cache) % 100 == 0:
            self._save_cache()
        
        return vec
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        to_compute = []
        indices = []
        
        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append(None)
                to_compute.append(text)
                indices.append(i)
        
        if to_compute:
            computed = self.backend.embed_batch(to_compute)
            for idx, vec in zip(indices, computed):
                results[idx] = vec
                key = hashlib.md5(texts[idx].encode()).hexdigest()
                self.cache[key] = vec
            self._save_cache()
        
        return results
    
    @property
    def dim(self) -> int:
        return self.backend.dim


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    provider = EmbeddingProvider()
    
    texts = [
        "Les langues officielles du Cameroun sont le français et l'anglais",
        "Habari za mzunguko wa mwezi katika sayansi",
        "什么是量子纠缠？用简单的话解释",
        "Berapa banyak bahasa yang digunakan di Indonesia?",
    ]
    
    for text in texts:
        vec = provider.embed(text)
        print(f"'{text[:50]}...' -> {len(vec)}d vector")
        
    # Similarity test
    import math
    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0
    
    v1 = provider.embed("Les langues officielles du Cameroun sont le français et l'anglais")
    v2 = provider.embed("Berapa banyak bahasa yang digunakan di Indonesia?")
    v3 = provider.embed("什么是量子纠缠？用简单的话解释")
    
    print(f"\nSimilarity 'Cameroun' vs 'Indonesia': {cosine(v1, v2):.3f}")
    print(f"Similarity 'Cameroun' vs 'quantum': {cosine(v1, v3):.3f}")
    print(f"(Similar languages should score higher than unrelated topics)")
