#!/usr/bin/env python3
"""
embed_provider.py - Text Embedding for Mazemaker Adapter

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
import warnings
from pathlib import Path

# fastembed >= 0.7 emits a cosmetic UserWarning when the
# intfloat/multilingual-e5-large model switched from CLS to mean pooling.
# Mean pooling is the model's documented aggregation; the prior CLS path
# was a fastembed bug. We want the new (correct) behaviour, so silence
# the noise rather than pin to 0.5.1.
warnings.filterwarnings(
    "ignore",
    message=r".*now uses mean pooling instead of CLS embedding.*",
    category=UserWarning,
)

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
        import time as time_module
        
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
        
        # If target is CUDA, wait for sufficient GPU memory (no silent fallback)
        if device == 'cuda':
            # Cap the waiter at 30s: 120s was too long — if the GPU is full
            # for that long, it's not coming back during this session, and
            # the user is paying multi-minute startup latency before the
            # first prompt. EMBED_GPU_WAIT_S overrides for ops who want a
            # longer hold.
            gpu_wait_s = float(os.environ.get("EMBED_GPU_WAIT_S", "30"))
            self._wait_for_gpu_memory(min_free_mb=2000, timeout=gpu_wait_s, poll_interval=2)
        
        self.model = SentenceTransformer(model_path, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._original_device = device
        self._last_used = time.time()
        print(f"[embed-server] Ready: {self.model_name} ({self.dim}d) on {device}")
    
    def _wait_for_gpu_memory(self, min_free_mb: float = 2000, timeout: float = 120, poll_interval: float = 5):
        """Wait for sufficient GPU memory to be available.
        
        Policy: GPU FIRST — if CUDA was chosen as target, wait for memory.
        Do NOT fall back to CPU silently.
        """
        import torch
        import time as time_module
        
        deadline = time_module.time() + timeout
        attempt = 0
        
        while time_module.time() < deadline:
            attempt += 1
            free = torch.cuda.mem_get_info(0)[0] / 1024**2
            if free >= min_free_mb:
                if attempt > 1:
                    print(f"[embed-server] GPU memory sufficient: {free:.0f} MB (after {attempt} attempts)")
                return
            if attempt <= 3:
                print(f"[embed-server] WARNING: Only {free:.0f}MB free GPU memory, need {min_free_mb}MB+ (attempt {attempt}, waiting {poll_interval}s...)", file=sys.stderr)
            time_module.sleep(poll_interval)
        
        # Timeout — GPU genuinely unavailable
        print(f"[embed-server] FATAL: GPU memory timeout ({free:.0f}MB < {min_free_mb}MB after {timeout}s)", file=sys.stderr)
        print(f"[embed-server] Policy: NO FALLBACK — GPU was target, not falling back to CPU", file=sys.stderr)
        raise RuntimeError(f"GPU memory insufficient after {timeout}s: {free:.0f}MB < {min_free_mb}MB")
    
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
        # Only reload GPU for actual embedding work, not status/ping/eject
        if cmd in ("embed", "embed_batch"):
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
    
    def _ensure_on_device(self, timeout: float = 60.0, poll_interval: float = 2.0):
        """Ensure model is on GPU, waiting if necessary. Raises on failure.
        
        Policy: GPU FIRST — wait for GPU, never silently fall back to CPU.
        This runs in the shared server process and must enforce the same
        no-fallback rule to prevent CUDA crashes from silent degradation.
        """
        import torch
        import time as time_module
        
        if self._original_device != 'cuda':
            return True
        
        current = next(self.model.parameters()).device
        if current.type != 'cpu':
            return True  # Already on GPU
        
        # Model is on CPU but CUDA was intended — wait and retry
        deadline = time_module.time() + timeout
        attempt = 0
        last_error = None
        
        while time_module.time() < deadline:
            attempt += 1
            try:
                free = torch.cuda.mem_get_info(0)[0] / 1024**2
                if free < 500:
                    last_error = f"GPU memory critically low: {free:.0f}MB"
                    if attempt <= 3:
                        print(f"[embed-server] WARNING: {last_error}, waiting... ({attempt})", file=sys.stderr)
                    time_module.sleep(poll_interval)
                    continue
                
                self.model.to('cuda')
                torch.cuda.synchronize()
                print(f"[embed-server] Reloaded to GPU after {attempt} attempt(s)")
                return True
                
            except Exception as e:
                last_error = str(e)
                if attempt <= 3:
                    print(f"[embed-server] GPU reload attempt {attempt} failed: {last_error}, retrying...", file=sys.stderr)
                time_module.sleep(poll_interval)
        
        # Timeout — GPU genuinely unavailable
        print(f"[embed-server] FATAL: GPU reload timed out after {attempt} attempts", file=sys.stderr)
        print(f"[embed-server] Policy: NO FALLBACK — GPU was intended, not falling back to CPU", file=sys.stderr)
        raise RuntimeError(f"GPU unavailable after {attempt} attempts: {last_error}")
    
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
    """Client that connects to SharedEmbedServer via UNIX socket.
    
    Retry logic: if server is temporarily unresponsive (e.g. loading model,
    processing another client), retry with exponential backoff before giving up.
    This prevents sessions from falling back to direct-load when the shared
    server is the intended architecture.
    """
    
    _MAX_RETRIES = 5
    _BASE_DELAY = 0.2  # seconds
    
    def __init__(self, timeout=10.0):
        self._sock = None
        self._dim = None
        self._timeout = timeout
        self._connect()
    
    def _connect(self):
        last_err = None
        for attempt in range(self._MAX_RETRIES):
            try:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.settimeout(self._timeout)
                self._sock.connect(str(SOCKET_PATH))
                # Get dim
                resp = self._send({"cmd": "ping"})
                self._dim = resp.get("dim", 1024)
                return  # success
            except (FileNotFoundError, ConnectionRefusedError) as e:
                # Hard \"server not there\" signals — no server is listening on
                # the socket path or the socket file is missing entirely.
                # Retrying just adds startup latency (the previous code spent
                # ~6s of backoff before giving up on a clearly-dead server).
                # Fail fast so _auto_detect falls through to a direct load.
                if self._sock:
                    try: self._sock.close()
                    except Exception: pass
                    self._sock = None
                raise e
            except (socket.timeout, socket.error, OSError) as e:
                last_err = e
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._BASE_DELAY * (2 ** attempt)  # 0.2, 0.4, 0.8, 1.6, 3.2s
                    time.sleep(delay)
        # All retries exhausted — raise last error so _auto_detect falls through
        raise last_err if last_err else RuntimeError("SharedEmbedClient: all retries failed")
    
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
    
    def _reconnect(self):
        """Reconnect if socket was broken (e.g. server restarted)."""
        try:
            self._sock.close()
        except Exception:
            pass
        self._connect()
    
    def embed(self, text):
        try:
            resp = self._send({"cmd": "embed", "text": text})
            return resp["vec"]
        except (socket.timeout, socket.error, OSError):
            # Socket may have broken — try reconnect + one retry
            try:
                self._reconnect()
                resp = self._send({"cmd": "embed", "text": text})
                return resp["vec"]
            except Exception:
                raise RuntimeError("SharedEmbedClient: embed failed after reconnect")
    
    def embed_batch(self, texts):
        try:
            resp = self._send({"cmd": "embed_batch", "texts": texts})
            return resp["vecs"]
        except (socket.timeout, socket.error, OSError):
            # Socket may have broken — try reconnect + one retry
            try:
                self._reconnect()
                resp = self._send({"cmd": "embed_batch", "texts": texts})
                return resp["vecs"]
            except Exception:
                raise RuntimeError("SharedEmbedClient: embed_batch failed after reconnect")
    
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
    # Initialised at class-body time. Previously lazy via
    # `if _lock is None: _lock = Lock()`, which has the same lock-init
    # race fixed in FastEmbedBackend (see iter 40 commit message). Eager
    # init eliminates the race; the cost is negligible (one Lock object
    # per backend class at module load).
    _lock = threading.Lock()
    
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
        import time as time_module

        # Lock is now class-level (initialised at class-body time, see iter
        # 40). The previous `if _lock is None: _lock = Lock()` lazy-init was
        # itself racy — two threads could install distinct Lock objects.

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
        
        # GPU-FIRST: if CUDA was chosen, wait for memory (no silent fallback).
        # Capped at EMBED_GPU_WAIT_S (default 30s) — see SharedEmbedServer
        # comment above for rationale.
        if device == 'cuda':
            gpu_wait_s = float(os.environ.get("EMBED_GPU_WAIT_S", "30"))
            _wait_for_gpu(self, min_free_mb=2000, timeout=gpu_wait_s, poll_interval=2)
        
        print(f"[embed] Loading {model_path} directly on {device}...")
        self.model = SentenceTransformer(model_path, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        SentenceTransformerBackend._shared_model = self.model
        SentenceTransformerBackend._shared_dim = self.dim
        SentenceTransformerBackend._shared_device = device
        print(f"[embed] {self.MODEL_NAME} ready ({self.dim}d)")

    def embed(self, text: str) -> list[float]:
        if self._is_client:
            try:
                return self._client.embed(text)
            except (socket.timeout, OSError, RuntimeError) as e:
                print(f"[embed] Server timeout ({e}), reconnecting...", file=sys.stderr)
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = SharedEmbedClient(timeout=15.0)
                return self._client.embed(text)
        SentenceTransformerBackend._touch()
        SentenceTransformerBackend._ensure_on_device()
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._is_client:
            try:
                return self._client.embed_batch(texts)
            except (socket.timeout, OSError, RuntimeError) as e:
                print(f"[embed] Server timeout ({e}), reconnecting...", file=sys.stderr)
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = SharedEmbedClient(timeout=15.0)
                return self._client.embed_batch(texts)
        SentenceTransformerBackend._touch()
        SentenceTransformerBackend._ensure_on_device()
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    @classmethod
    def _touch(cls):
        """Record last use time for idle eject timer."""
        cls._last_used = time.time()

    @classmethod
    def _ensure_on_device(cls, timeout: float = 60.0, poll_interval: float = 2.0):
        import torch
        import time as time_module

        if cls._shared_device != 'cuda':
            return
        if cls._shared_model is None:
            return
        current = next(cls._shared_model.parameters()).device
        if current.type != 'cpu':
            return

        deadline = time_module.time() + timeout
        attempt = 0
        last_error = None
        while time_module.time() < deadline:
            attempt += 1
            try:
                free = torch.cuda.mem_get_info(0)[0] / 1024**2
                if free < 500:
                    last_error = f"GPU memory critically low: {free:.0f}MB"
                    if attempt == 1:
                        print(f"[embed] WARNING: {last_error}, waiting... ({attempt})", file=sys.stderr)
                    time_module.sleep(poll_interval)
                    continue
                cls._shared_model.to('cuda')
                torch.cuda.synchronize()
                print(f"[embed] Reloaded to GPU after {attempt} attempt(s)")
                return
            except Exception as e:
                last_error = str(e)
                if attempt <= 3:
                    print(f"[embed] GPU reload attempt {attempt} failed: {last_error}, retrying...", file=sys.stderr)
                time_module.sleep(poll_interval)
        print(f"[embed] FATAL: GPU reload timed out after {attempt} attempts. Last error: {last_error}", file=sys.stderr)
        raise RuntimeError(f"GPU unavailable after {attempt} attempts: {last_error}")

    @classmethod
    def eject(cls):
        """Manually eject model to CPU (frees GPU memory now)."""
        try:
            client = SharedEmbedClient()
            client._send({"cmd": "eject"})
            client.close()
            return
        except Exception:
            pass
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
        try:
            client = SharedEmbedClient()
            resp = client._send({"cmd": "status"})
            client.close()
            resp["mode"] = "shared"
            return resp
        except Exception:
            pass
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
        except Exception:
            return {"loaded": True, "mode": "direct", "error": "could not determine status"}


def _wait_for_gpu(min_free_mb: float = 2000, timeout: float = 120, poll_interval: float = 5):
    """Wait for sufficient GPU memory. Raises on timeout (no fallback)."""
    import torch
    import time as time_module

    deadline = time_module.time() + timeout
    attempt = 0
    while time_module.time() < deadline:
        attempt += 1
        free = torch.cuda.mem_get_info(0)[0] / 1024**2
        if free >= min_free_mb:
            if attempt > 1:
                print(f"[embed] GPU memory sufficient: {free:.0f} MB (after {attempt} attempts)")
            return
        if attempt <= 3:
            print(f"[embed] WARNING: Only {free:.0f}MB free GPU memory, need {min_free_mb}MB+ (attempt {attempt}, waiting {poll_interval}s...)", file=sys.stderr)
        time_module.sleep(poll_interval)
    print(f"[embed] FATAL: GPU memory timeout ({free:.0f}MB < {min_free_mb}MB after {timeout}s)", file=sys.stderr)
    raise RuntimeError(f"GPU memory insufficient after {timeout}s: {free:.0f}MB < {min_free_mb}MB")


class TfidfSvdBackend:
    """Pure numpy TF-IDF + SVD embedding (no ML dependencies)."""

    STATE_FILE = CACHE_DIR / "tfidf_state.npz"

    def __init__(self, dim: int = DIMENSION):
        import numpy as np
        self.np = np
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None
        self._trained = False
        self._corpus: list[str] = []
        self._load_state()

    def _load_state(self):
        """Load a previously-fitted vocab/IDF/SVD from disk if dim matches."""
        if not self.STATE_FILE.exists():
            return
        try:
            np = self.np
            with np.load(self.STATE_FILE, allow_pickle=False) as data:
                if int(data["dim"]) != self.dim:
                    return
                vocab_words = data["vocab_words"]
                self.vocab = {str(w): i for i, w in enumerate(vocab_words)}
                self.idf = data["idf"]
                self.svd_components = data["svd_components"]
            self._trained = True
        except (KeyError, ValueError, OSError):
            self._trained = False
            self.vocab = {}
            self.idf = None
            self.svd_components = None

    def _save_state(self):
        """Persist vocab/IDF/SVD to disk so future starts skip the fit.

        Writes to a sibling .tmp file and os.replace()'s onto STATE_FILE so
        an interrupted save (Ctrl+C, OOM, kill -9) can never leave a half-
        written .npz on disk that the next process would have to retrain
        from scratch (~30-60s SVD fit).
        """
        if not self._trained or self.svd_components is None or self.idf is None:
            return
        tmp_path = None
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np = self.np
            words = sorted(self.vocab.items(), key=lambda kv: kv[1])
            # Use a fixed-width unicode dtype so the array is pickle-free —
            # _load_state() uses allow_pickle=False to keep the cache file
            # safe to read even if it's tampered with.
            vocab_words = np.array([w for w, _ in words], dtype="U")
            # np.savez auto-appends .npz when given a string/Path lacking that
            # suffix, which would land us at <name>.tmp.npz instead of
            # <name>.tmp. Pass a file handle so the path is honoured exactly.
            tmp_path = self.STATE_FILE.with_name(self.STATE_FILE.name + ".tmp")
            with open(tmp_path, "wb") as fh:
                np.savez(
                    fh,
                    dim=np.int64(self.dim),
                    vocab_words=vocab_words,
                    idf=self.idf,
                    svd_components=self.svd_components,
                )
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.STATE_FILE)
            tmp_path = None  # ownership transferred
        except OSError:
            pass
        finally:
            # If replace() didn't run (exception before it), clean up the tmp.
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    
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
        
        # SVD for dimensionality reduction.  The previous implementation
        # called np.linalg.svd on a (n_docs, 5000) dense matrix despite the
        # "Randomized SVD for speed" comment — that's full deterministic SVD,
        # O(n·m²), which dominated cold-start (30-60s on a real corpus).
        # The randomized variant below is O(n·m·k) with k=self.dim, typically
        # 5-10x faster while preserving the top-k singular subspace to within
        # tens of basis points.
        if vocab_size > self.dim:
            sub = tfidf[:, :min(vocab_size, 5000)]
            self.svd_components = self._randomized_svd_components(sub, self.dim)
        else:
            self.svd_components = np.eye(vocab_size, self.dim)

        self._trained = True
        self._save_state()

    def _randomized_svd_components(self, M, k: int, n_oversamples: int = 10, n_iter: int = 4):
        """Compute the top-k right singular vectors of M via randomized SVD.

        Returns Vt[:k].T with shape (M.shape[1], k) — same orientation the
        full-SVD code path produced.  See Halko/Martinsson/Tropp 2011.
        """
        np = self.np
        n_rows, n_cols = M.shape
        ell = min(k + n_oversamples, n_cols)
        rng = np.random.default_rng(0xBADC0DE)
        Omega = rng.standard_normal((n_cols, ell)).astype(M.dtype, copy=False)
        Y = M @ Omega
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(Y)
            Z = M.T @ Q
            Q2, _ = np.linalg.qr(Z)
            Y = M @ Q2
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ M
        _, _, Vt = np.linalg.svd(B, full_matrices=False)
        return Vt[:k].T
    
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


class FastEmbedBackend:
    """FastEmbed ONNX backend — primary embedding backend.

    Uses intfloat/multilingual-e5-large (1024d) via ONNX runtime.
    No PyTorch dependency. ~50ms per embedding on CPU.

    SINGLETON: Model is loaded ONCE and shared across all instances.
    This prevents multiple ONNX model copies from being loaded (each ~1GB).
    """
    MODEL_NAME = "intfloat/multilingual-e5-large"

    # Class-level singleton — shared across all instances
    _shared_model = None
    _shared_dim = None
    # Initialised at class-body time so the double-checked-locking idiom in
    # __init__ is actually safe. Previously the lock itself was lazy-inited
    # under `if _lock is None: _lock = Lock()`, which is not atomic — two
    # threads racing through the first construction could each install a
    # different Lock object, defeating the singleton-load guarantee and
    # producing two ONNX model copies (the exact failure the singleton was
    # meant to prevent — see commit ac02e8b).
    _lock = threading.Lock()

    def __init__(self, dim: int = DIMENSION):
        # Try to reuse shared model — fast-path without taking the lock.
        if FastEmbedBackend._shared_model is not None:
            self._model = FastEmbedBackend._shared_model
            self.dim = FastEmbedBackend._shared_dim or dim
            return

        with FastEmbedBackend._lock:
            # Double-check after acquiring lock
            if FastEmbedBackend._shared_model is not None:
                self._model = FastEmbedBackend._shared_model
                self.dim = FastEmbedBackend._shared_dim or dim
                return

            self.dim = dim
            self._model = None
            self._load_direct()

    def _load_direct(self):
        """Load model directly — called once under lock."""
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.MODEL_NAME)
            # Verify dimension
            test = list(self._model.embed(["test"]))
            self.dim = len(test[0])

            # Share the model with all future instances
            FastEmbedBackend._shared_model = self._model
            FastEmbedBackend._shared_dim = self.dim

            print(f"[embed] FastEmbed loaded: {self.MODEL_NAME} ({self.dim}d)")
        except ImportError:
            print("[embed] fastembed not installed — pip install fastembed", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[embed] FastEmbed load failed: {e}", file=sys.stderr)
            raise
    
    def embed(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("FastEmbed model not loaded")
        result = list(self._model.embed([text]))
        return result[0].tolist() if hasattr(result[0], 'tolist') else list(result[0])
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("FastEmbed model not loaded")
        results = list(self._model.embed(texts))
        return [r.tolist() if hasattr(r, 'tolist') else list(r) for r in results]


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
    # Bounded LRU cap. ~8KB per 1024-d vector → 32MB at 4096 entries.
    # Override with EMBED_CACHE_MAX env var.
    _CACHE_MAX_DEFAULT = 4096

    def __init__(self, backend: str = "auto"):
        from collections import OrderedDict
        self._cache_max = max(64, int(os.environ.get("EMBED_CACHE_MAX",
                                                     self._CACHE_MAX_DEFAULT)))
        self.cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._load_cache()

        # EMBED_BACKEND env always wins, regardless of what the caller passed.
        # Some callers (like the v2 mcp container) pass placeholder strings
        # ("http") expecting their own dispatch layer; the env lets the
        # operator force a specific in-process backend without code changes.
        forced = os.environ.get('EMBED_BACKEND', '').strip().lower()
        if forced in ('sentence-transformers', 'st', 'sbert'):
            self.backend = SentenceTransformerBackend()
            print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d) [forced via EMBED_BACKEND={forced}]")
            return
        if forced == 'fastembed':
            self.backend = FastEmbedBackend()
            print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d) [forced via EMBED_BACKEND={forced}]")
            return

        if backend == "auto":
            self.backend = self._auto_detect()
        elif backend == "fastembed":
            self.backend = FastEmbedBackend()
        elif backend == "sentence-transformers":
            self.backend = SentenceTransformerBackend()
        elif backend == "tfidf":
            self.backend = TfidfSvdBackend()
        elif backend == "hash":
            self.backend = HashBackend()
        else:
            # Try as fastembed first, fall back to hash
            try:
                self.backend = FastEmbedBackend()
            except Exception:
                self.backend = HashBackend()
        
        print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d)")
    
    def _auto_detect(self):
        """Auto-detect best available backend.

        Priority: Shared Server (already running) > CUDA sentence-transformers > FastEmbed >
                  MPS sentence-transformers > CPU sentence-transformers > TF-IDF > hash

        The shared server is checked FIRST because it means another process already loaded
        the model. We do NOT reload it — we connect via socket.

        Override via env EMBED_BACKEND={sentence-transformers,fastembed,hash}
        — needed when the auto-pick conflicts with stored vectors (e.g. a DB
        embedded under sentence-transformers can't be queried by FastEmbed even
        though both are 1024d, since the vector spaces differ).
        """
        try:
            import torch
        except ImportError:
            torch = None

        # -1. EXPLICIT OVERRIDE via env — bypasses the auto-select chain
        forced = os.environ.get('EMBED_BACKEND', '').strip().lower()
        if forced in ('sentence-transformers', 'st', 'sbert'):
            backend = SentenceTransformerBackend()
            print(f"[embed] Forced (env EMBED_BACKEND={forced}): sentence-transformers ({backend.dim}d)")
            return backend
        if forced == 'fastembed':
            backend = FastEmbedBackend()
            print(f"[embed] Forced (env EMBED_BACKEND={forced}): FastEmbed ({backend.dim}d)")
            return backend

        # 0. SHARED SERVER FIRST — if socket exists and server is alive, USE IT
        # This is the whole point of the shared server architecture!
        if not os.environ.get('EMBED_NO_SHARED'):
            try:
                client = SharedEmbedClient()
                dim = client.dim
                client.close()
                backend = SentenceTransformerBackend()
                backend._is_client = True
                backend.dim = dim
                backend.model = None
                backend._client = SharedEmbedClient()
                print(f"[embed] Auto-selected: shared server ({dim}d)")
                return backend
            except Exception:
                pass  # No server running, fall through to direct load

        # 1. CUDA sentence-transformers (user's RTX 4060 Ti has 15GB VRAM)
        # Policy: GPU FIRST — if CUDA is available and we attempt it, do NOT fall back
        # to FastEmbed/CPU on failure. Either GPU succeeds or we raise.
        if torch and torch.cuda.is_available():
            try:
                free = torch.cuda.mem_get_info(0)[0] / 1024**2
                if free > 2000:  # At least 2GB VRAM for batching
                    backend = SentenceTransformerBackend()
                    # SentenceTransformerBackend() may have started the shared server
                    # and connected as a client. If so, the model is already loaded in
                    # the server process — return immediately without loading it again.
                    if backend._is_client:
                        print(f"[embed] Auto-selected: sentence-transformers CUDA ({backend.dim}d, {free:.0f}MB free)")
                        return backend
                    backend._is_client = False
                    # Force CUDA load directly
                    backend._load_direct = lambda: None  # prevent recursion
                    # Actually load on CUDA
                    from sentence_transformers import SentenceTransformer
                    safe_name = backend.MODEL_NAME.replace('/', '--')
                    cache_base = MODEL_DIR / f"models--{safe_name}"
                    model_path = None
                    refs_main = cache_base / "refs" / "main"
                    if refs_main.exists():
                        snap_hash = refs_main.read_text().strip()
                        sp = cache_base / "snapshots" / snap_hash
                        if (sp / "config.json").exists():
                            model_path = str(sp)
                    if model_path is None:
                        snapshots_dir = cache_base / "snapshots"
                        if snapshots_dir.exists():
                            for snap in snapshots_dir.iterdir():
                                if (snap / "config.json").exists():
                                    model_path = str(snap)
                                    break
                    if model_path:
                        backend.model = SentenceTransformer(model_path, device='cuda')
                        backend.dim = backend.model.get_sentence_embedding_dimension()
                        SentenceTransformerBackend._shared_model = backend.model
                        SentenceTransformerBackend._shared_dim = backend.dim
                        SentenceTransformerBackend._shared_device = 'cuda'
                        print(f"[embed] Auto-selected: sentence-transformers CUDA ({backend.dim}d, {free:.0f}MB free)")
                        return backend
            except (ImportError, Exception) as e:
                # CUDA was available and we attempted it — GPU was the target.
                # Do NOT fall back to FastEmbed/CPU. Raise immediately.
                print(f"[embed] CUDA sentence-transformers FAILED (GPU was target, not falling back): {e}", file=sys.stderr)
                raise RuntimeError(f"CUDA embedding failed and no fallback allowed by policy: {e}") from e

        # Below this line: CUDA was NOT available — these are legitimate fallbacks
        # for systems WITHOUT GPU access. NOT for systems that chose GPU and failed.

        # 2. FastEmbed ONNX (CPU, no PyTorch dependency, fast)
        try:
            backend = FastEmbedBackend()
            print(f"[embed] Auto-selected: FastEmbed ({backend.dim}d)")
            return backend
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"[embed] FastEmbed failed: {e}", file=sys.stderr)

        # 3. MPS sentence-transformers (Apple Silicon)
        if torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                backend = SentenceTransformerBackend()
                print("[embed] Auto-selected: sentence-transformers MPS")
                return backend
            except (ImportError, Exception) as e:
                if not isinstance(e, ImportError):
                    print(f"[embed] MPS sentence-transformers failed: {e}", file=sys.stderr)

        # 4. CPU sentence-transformers
        try:
            import sentence_transformers
            backend = SentenceTransformerBackend()
            print("[embed] Auto-selected: sentence-transformers CPU")
            return backend
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"[embed] CPU sentence-transformers failed: {e}", file=sys.stderr)

        # 5. Hash fallback — instant, deterministic, zero dependencies.
        # Preferred over TF-IDF+SVD because the latter requires a corpus fit
        # (full SVD on a dense matrix) that costs 30-60s on cold start with
        # no semantic-quality win that justifies the wait.  TF-IDF is still
        # available via EMBED_BACKEND=tfidf for users who explicitly want it.
        print("[embed] Auto-selected: hash (zero dependencies)")
        return HashBackend()
    
    def _load_cache(self):
        from collections import OrderedDict
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, OrderedDict):
                    self.cache = raw
                elif isinstance(raw, dict):
                    self.cache = OrderedDict(raw.items())
                else:
                    self.cache = OrderedDict()
            except Exception:
                self.cache = OrderedDict()
            # On load, evict back down to the cap if a previous (unbounded)
            # session left a giant pickle on disk.
            self._evict_to_cap()

    def _save_cache(self):
        """Persist cache atomically so a crash mid-write can't corrupt it."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".tmp")
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(self.cache, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, CACHE_FILE)
        except OSError:
            try: tmp.unlink()
            except OSError: pass

    def _evict_to_cap(self) -> None:
        while len(self.cache) > self._cache_max:
            self.cache.popitem(last=False)  # FIFO: drop oldest insertion

    def _record(self, key: str, vec: list[float]) -> None:
        """Insert / refresh-LRU a cache entry under the bound."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = vec
            return
        self.cache[key] = vec
        if len(self.cache) > self._cache_max:
            self._evict_to_cap()

    def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            # LRU touch
            self.cache.move_to_end(key)
            return self.cache[key]

        vec = self.backend.embed(text)
        self._record(key, vec)

        # Save periodically (every 100 *misses*, not every 100 entries — the
        # old `len(cache) % 100 == 0` would re-save constantly once the cache
        # plateaued at a multiple of 100, since len was unchanged after each
        # eviction-replace cycle).
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
                self.cache.move_to_end(key)
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
                self._record(key, vec)
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
