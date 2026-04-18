#!/usr/bin/env python3
"""
embed_provider.py - Text Embedding for Neural Memory Adapter

SHARED MODE (default): First process starts a UNIX socket server holding
the model. All other processes connect as clients. ONE model instance
for ALL hermes sessions. Smart eject after 20s idle.

FALLBACK: If shared server can't start, loads model directly per-process.

Env vars:
  EMBED_MODEL        — model name (default: BAAI/bge-m3)
  EMBED_IDLE_TIMEOUT — seconds before GPU→CPU eject (default: 20)
  EMBED_DEVICE       — force device (cuda/cpu/mps, default: auto)
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
DIMENSION = 1024  # BAAI/bge-m3 output dim

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
    
    def __init__(self, model_name=None, device=None, idle_timeout=20):
        self.model_name = model_name or os.environ.get('EMBED_MODEL', 'BAAI/bge-m3')
        self.idle_timeout = idle_timeout
        self.model = None
        self.dim = None
        self.device = device
        self._last_used = 0.0
        self._original_device = None
        self._lock = threading.Lock()
        self._running = False
        self._sock = None
    
    def start(self):
        """Load model and start listening. Returns True if started, False if already running."""
        if SOCKET_PATH.exists():
            # Check if existing server is alive
            try:
                client = SharedEmbedClient()
                if client.ping():
                    print(f"[embed-server] Already running at {SOCKET_PATH}")
                    return False
            except:
                pass
            # Stale socket
            SOCKET_PATH.unlink()
        
        self._load_model()
        self._start_listener()
        self._start_eject_timer()
        print(f"[embed-server] Listening at {SOCKET_PATH}")
        return True
    
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        import torch
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        device = 'cpu'
        if self.device:
            device = self.device
        elif torch.cuda.is_available():
            free = torch.cuda.mem_get_info(0)[0] / 1024**2
            if free > 500:
                device = 'cuda'
                print(f"[embed-server] CUDA: {free:.0f} MB free")
        
        if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        
        # Use snapshot path directly (works offline, avoids HF hub issues)
        safe_name = self.model_name.replace('/', '--')
        cache_base = MODEL_DIR / f"models--{safe_name}"
        refs_main = cache_base / "refs" / "main"
        
        model_path = None
        if refs_main.exists():
            snapshot_hash = refs_main.read_text().strip()
            snapshot_path = cache_base / "snapshots" / snapshot_hash
            if (snapshot_path / "config.json").exists():
                model_path = str(snapshot_path)
        
        if model_path is None:
            # Fallback: find any snapshot with config.json
            snapshots_dir = cache_base / "snapshots"
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    if (snap / "config.json").exists():
                        model_path = str(snap)
                        break
        
        if model_path is None:
            print(f"[embed-server] ERROR: No cached model found at {cache_base}", file=sys.stderr)
            raise FileNotFoundError(f"No cached model: {self.model_name}")
        
        print(f"[embed-server] Loading {model_path} on {device}...")
        self.model = SentenceTransformer(model_path, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._original_device = device
        self._last_used = time.time()
        print(f"[embed-server] Ready: {self.model_name} ({self.dim}d) on {device}")
    
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
                # Read length-prefixed message
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
        self._ensure_on_device()
        
        with self._lock:
            try:
                if cmd == "embed":
                    vec = self.model.encode(req["text"], normalize_embeddings=True)
                    return {"ok": True, "vec": vec.tolist()}
                elif cmd == "embed_batch":
                    vecs = self.model.encode(req["texts"], normalize_embeddings=True, show_progress_bar=False)
                    return {"ok": True, "vecs": [v.tolist() for v in vecs]}
                elif cmd == "status":
                    device = next(self.model.parameters()).device
                    return {
                        "ok": True, "model": self.model_name, "dim": self.dim,
                        "device": str(device), "original": self._original_device,
                        "idle": round(time.time() - self._last_used, 1),
                        "timeout": self.idle_timeout,
                    }
                elif cmd == "ping":
                    return {"ok": True, "dim": self.dim}
                elif cmd == "eject":
                    self._eject_to_cpu()
                    return {"ok": True}
                else:
                    return {"ok": False, "error": f"unknown cmd: {cmd}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}
    
    def _ensure_on_device(self):
        if self._original_device == 'cpu':
            return
        try:
            import torch
            current = next(self.model.parameters()).device
            if current.type == 'cpu' and self._original_device == 'cuda':
                self.model.to('cuda')
            elif current.type == 'cpu' and self._original_device == 'mps':
                self.model.to('mps')
        except Exception:
            pass
    
    def _eject_to_cpu(self):
        try:
            import torch
            current = next(self.model.parameters()).device
            if current.type == 'cpu':
                return
            self.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[embed-server] Ejected to CPU (freed GPU)")
        except Exception as e:
            print(f"[embed-server] Eject failed: {e}", file=sys.stderr)
    
    def _start_eject_timer(self):
        if self.idle_timeout <= 0:
            return
        def _check():
            while self._running:
                time.sleep(5)  # Check every 5s for fast eject
                idle = time.time() - self._last_used
                if idle > self.idle_timeout:
                    self._eject_to_cpu()
        t = threading.Thread(target=_check, daemon=True, name="embed-eject")
        t.start()
    
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
    
    def __init__(self):
        self._sock = None
        self._dim = None
        self._connect()
    
    def _connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(SOCKET_PATH))
        # Get dim
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
    
    def close(self):
        if self._sock:
            self._sock.close()


# ============================================================================
# SentenceTransformerBackend (with shared server support)
# ============================================================================

class SentenceTransformerBackend:
    """Uses sentence-transformers (default: BAAI/bge-large-en-v1.5, 1024d)
    
    Singleton: model loaded once and shared across all instances.
    Cached locally at ~/.neural_memory/models/.
    
    SMART EJECT: After EMBED_IDLE_TIMEOUT seconds of inactivity, model moves
    to CPU to free GPU memory. Automatically reloads on next embed() call.
    
    Env vars:
      EMBED_MODEL — override model name (default: BAAI/bge-m3)
      EMBED_IDLE_TIMEOUT — seconds before eject to CPU (default: 300, 0=disabled)
      EMBED_DEVICE — force device (cuda/cpu/mps, default: auto)
    """
    MODEL_NAME = os.environ.get('EMBED_MODEL', os.environ.get('SENTENCE_TRANSFORMER_MODEL', 'BAAI/bge-m3'))
    IDLE_TIMEOUT = int(os.environ.get('EMBED_IDLE_TIMEOUT', '20'))
    FORCED_DEVICE = os.environ.get('EMBED_DEVICE', None)
    
    _shared_model = None
    _shared_dim = None
    _shared_device = None
    _last_used = 0.0
    _eject_timer = None
    _lock = None  # threading.Lock, lazy init
    
    def __init__(self):
        # Try shared server first (unless disabled)
        if not os.environ.get('EMBED_NO_SHARED'):
            try:
                self._client = SharedEmbedClient()
                self.dim = self._client.dim
                self.model = None  # Not loaded locally
                self._is_client = True
                print(f"[embed] Connected to shared server ({self.dim}d)")
                return
            except Exception:
                pass  # No server running, start one or load directly
        
        self._is_client = False
        self._client = None
        
        # Try to become the server
        if not os.environ.get('EMBED_NO_SHARED'):
            server = SharedEmbedServer(
                model_name=self.MODEL_NAME,
                device=self.FORCED_DEVICE,
                idle_timeout=self.IDLE_TIMEOUT,
            )
            if server.start():
                # We're the server — also connect as client for embed calls
                try:
                    self._client = SharedEmbedClient()
                    self.dim = self._client.dim
                    self.model = None
                    self._is_client = True
                    print(f"[embed] Started shared server, using client mode")
                    return
                except:
                    pass
        
        # Fallback: load model directly (old behavior)
        self._load_direct()
    
    def _load_direct(self):
        """Fallback: load model directly into this process (original behavior)."""
        from sentence_transformers import SentenceTransformer
        import torch
        import threading
        
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
        if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        
        # Use snapshot path directly (works offline)
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
        SentenceTransformerBackend._shared_device = device
        print(f"[embed] {self.MODEL_NAME} ready ({self.dim}d)")
    
    def embed(self, text: str) -> list[float]:
        if self._is_client:
            return self._client.embed(text)
        SentenceTransformerBackend._touch()
        SentenceTransformerBackend._ensure_on_device()
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._is_client:
            return self._client.embed_batch(texts)
        SentenceTransformerBackend._touch()
        SentenceTransformerBackend._ensure_on_device()
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    
    @classmethod
    def eject(cls):
        """Manually eject model to CPU (frees GPU memory now)."""
        # Try shared server eject
        try:
            client = SharedEmbedClient()
            client._send({"cmd": "eject"})
            client.close()
            return
        except:
            pass
        # Direct model eject
        if cls._shared_model is None:
            return
        try:
            import torch
            device = next(cls._shared_model.parameters()).device
            if device.type != 'cpu':
                cls._shared_model.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[embed] Ejected to CPU")
        except Exception as e:
            print(f"[embed] Eject failed: {e}", file=sys.stderr)
    
    @classmethod
    def status(cls):
        """Return model status dict."""
        # Try shared server status
        try:
            client = SharedEmbedClient()
            resp = client._send({"cmd": "status"})
            client.close()
            resp["mode"] = "shared"
            return resp
        except:
            pass
        
        # Direct model status
        if cls._shared_model is None:
            return {"loaded": False, "mode": "none"}
        try:
            device = next(cls._shared_model.parameters()).device
            idle = time.time() - cls._last_used
            return {
                "loaded": True, "mode": "direct",
                "model": cls.MODEL_NAME,
                "dim": cls._shared_dim,
                "device": str(device),
                "original_device": cls._shared_device,
                "idle_seconds": round(idle, 1),
                "eject_timeout": cls.IDLE_TIMEOUT,
                "ejected": device.type == 'cpu' and cls._shared_device != 'cpu',
            }
        except:
            return {"loaded": True, "mode": "direct", "error": "could not determine status"}


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
        """Hash-based fallback for pre-training"""
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
        """Fit on a corpus of texts"""
        np = self.np
        # Build vocabulary
        doc_freq = {}
        all_tokens = []
        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.append(list(tokens))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        
        # Keep top 10000 tokens
        sorted_vocab = sorted(doc_freq.items(), key=lambda x: -x[1])[:10000]
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_vocab)}
        vocab_size = len(self.vocab)
        
        # IDF
        n_docs = len(texts)
        self.idf = np.zeros(vocab_size)
        for word, idx in self.vocab.items():
            self.idf[idx] = np.log((n_docs + 1) / (doc_freq.get(word, 1) + 1)) + 1
        
        # Build TF-IDF matrix
        tfidf = np.zeros((n_docs, vocab_size))
        for i, tokens in enumerate(all_tokens):
            for t in tokens:
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] += 1
            # Apply IDF
            tfidf[i] *= self.idf
            # Normalize
            norm = np.linalg.norm(tfidf[i])
            if norm > 0:
                tfidf[i] /= norm
        
        # SVD for dimensionality reduction
        if vocab_size > self.dim:
            # Randomized SVD for speed
            U, S, Vt = np.linalg.svd(tfidf[:, :min(vocab_size, 5000)], full_matrices=False)
            self.svd_components = Vt[:self.dim].T  # (vocab, dim)
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
        
        # Pad or truncate to dim
        result = np.zeros(self.dim)
        result[:min(len(vec), self.dim)] = vec[:self.dim]
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        
        return result.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Optimized batch embedding using vectorized operations."""
        np = self.np
        
        # If not trained yet, check if we have enough to train
        if not self._trained:
            self._corpus.extend(texts)
            if len(self._corpus) >= 5:
                self.fit(self._corpus)
            # Return hash embeddings for each text
            return [self._hash_embed(t) for t in texts]
        
        # Tokenize all texts at once
        all_tokens = [self._tokenize(t) for t in texts]
        n_texts = len(texts)
        vocab_size = len(self.vocab)
        
        # Build TF matrix for all texts at once
        tfidf = np.zeros((n_texts, vocab_size))
        for i, tokens in enumerate(all_tokens):
            for t in tokens:
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] += 1
        
        # Apply IDF
        if self.idf is not None:
            tfidf *= self.idf
        
        # Normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        tfidf /= norms
        
        # Apply SVD projection
        if self.svd_components is not None:
            result = tfidf @ self.svd_components
        else:
            result = tfidf
        
        # Pad or truncate to dim and normalize
        batch_result = np.zeros((n_texts, self.dim))
        copy_len = min(result.shape[1], self.dim) if result.ndim > 1 else min(len(result), self.dim)
        if result.ndim > 1:
            batch_result[:, :copy_len] = result[:, :copy_len]
        else:
            batch_result[0, :copy_len] = result[:copy_len]
        
        # Final normalization
        norms = np.linalg.norm(batch_result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        batch_result /= norms
        
        return [row.tolist() for row in batch_result]


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
            # Distribute across dimensions
            for j in range(3):  # 3 positions per token
                idx = (h ^ (j * 2654435761)) % self.dim
                vec[idx] += 1.0 / (1.0 + i * 0.1)  # Position decay
                h = (h >> 8) | ((h & 0xFF) << 24)
        
        # Normalize
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
        elif backend == "sentence-transformers":
            self.backend = SentenceTransformerBackend()
        elif backend == "tfidf":
            self.backend = TfidfSvdBackend()
        else:
            self.backend = HashBackend()
        
        print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d)")
    
    def _auto_detect(self):
        """Auto-detect best available backend.
        Priority: CUDA sentence-transformers > MPS sentence-transformers > CPU sentence-transformers > TF-IDF > hash
        """
        # Try sentence-transformers first (CUDA, MPS, or CPU)
        try:
            import sentence_transformers
            import torch
            if torch.cuda.is_available():
                device = "CUDA"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "MPS"
            else:
                device = "CPU"
            backend = SentenceTransformerBackend()
            print(f"[embed] Auto-selected: sentence-transformers ({device})")
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
