"""access_logger.py — JSONL recall-event log for production observability.

Minimal version of mazemaker's `access_logger.py` (mazemaker pass-2
Pattern D). Pure observability: records every recall call with query,
top-K IDs, latency, channel breakdown. NO co-occurrence pair extraction
(that half is for LSTM training pairs we don't run).

Usage:
    from access_logger import RecallAccessLogger
    logger = RecallAccessLogger()  # default path
    logger.log(query="...", k=5, results=[{"id": ...}, ...],
               latency_ms=42.0, channels=["semantic", "sparse"])

Default log path: ~/.neural_memory/logs/recall-access.jsonl
Auto-rotates at ~100MB to recall-access.jsonl.<N>.

Why this matters: once AE-builder's email/materials/dashboard data
flows in, we'll need to know WHICH queries Tito actually makes to tune
the substrate. Without this log, weight tuning is theoretical. With
it, we can mine N days of real usage for labeling candidates +
weight-tuning ground truth.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


class RecallAccessLogger:
    """Append-only JSONL log of recall events. Thread-safe."""

    DEFAULT_PATH = Path.home() / ".neural_memory" / "logs" / "recall-access.jsonl"
    ROTATE_AT_BYTES = 100 * 1024 * 1024  # 100 MB

    def __init__(self, path: Optional[Path] = None,
                 rotate_at_bytes: Optional[int] = None):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rotate_at = rotate_at_bytes or self.ROTATE_AT_BYTES
        self._lock = threading.Lock()

    def log(self, *,
            query: str,
            k: int,
            results: list[dict[str, Any]],
            latency_ms: float,
            channels: Optional[list[str]] = None,
            method: str = "hybrid_recall",
            kind_filter: Optional[str] = None,
            rerank: Optional[bool] = None) -> None:
        """Append one recall event. Best-effort — silent on filesystem errors
        so production retrieval is never blocked by logging."""
        try:
            entry = {
                "ts": int(time.time()),
                "method": method,
                "query": query[:500],  # cap query length to keep entries small
                "k": k,
                "n_results": len(results),
                "top_ids": [r.get("id") for r in results[:10]],
                "latency_ms": round(latency_ms, 2),
                "channels": channels or [],
                "kind": kind_filter,
                "rerank": rerank,
            }
            self._maybe_rotate()
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception:
            pass

    def _maybe_rotate(self) -> None:
        """Rotate log if it exceeds size threshold.

        Reviewer B1 fix 2026-05-02: size-check now inside lock so
        concurrent rotation can't split-write across files.
        """
        try:
            with self._lock:
                if not self.path.exists():
                    return
                size = self.path.stat().st_size
                if size < self.rotate_at:
                    return
                # Find next available rotation slot
                for n in range(1, 1000):
                    rotated = self.path.with_suffix(f".jsonl.{n}")
                    if not rotated.exists():
                        self.path.rename(rotated)
                        break
        except Exception:
            pass


# Singleton convenience — production callers can use this default.
# Set NM_DISABLE_ACCESS_LOG=1 in env to skip logging entirely.
_default_logger: Optional[RecallAccessLogger] = None


def default_logger() -> Optional[RecallAccessLogger]:
    """Lazy-init the default singleton. None if env disables it."""
    global _default_logger
    if os.environ.get("NM_DISABLE_ACCESS_LOG", "") == "1":
        return None
    if _default_logger is None:
        _default_logger = RecallAccessLogger()
    return _default_logger
