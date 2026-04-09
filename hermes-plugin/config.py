"""Configuration for Neural Memory plugin.

Defaults:
  DB path:               ~/.neural_memory/hermes.db
  Embedding backend:     auto (sentence-transformers > tfidf > hash)
  Consolidation interval: 300s
  Max episodic memories:  50000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = str(Path.home() / ".neural_memory" / "hermes.db")
DEFAULT_EMBEDDING_BACKEND = "auto"          # auto | sentence-transformers | tfidf | hash
DEFAULT_CONSOLIDATION_INTERVAL = 300        # seconds
DEFAULT_MAX_EPISODIC = 50000
DEFAULT_SIMILARITY_THRESHOLD = 0.15         # auto-connect threshold
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_PREFETCH_LIMIT = 5


def get_config() -> Dict[str, Any]:
    """Read neural memory config from ~/.hermes/config.yaml.

    Falls back to defaults if the file doesn't exist or the section is missing.
    """
    config = {
        "db_path": DEFAULT_DB_PATH,
        "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        "consolidation_interval": DEFAULT_CONSOLIDATION_INTERVAL,
        "max_episodic": DEFAULT_MAX_EPISODIC,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "search_limit": DEFAULT_SEARCH_LIMIT,
        "prefetch_limit": DEFAULT_PREFETCH_LIMIT,
    }

    try:
        from hermes_cli.config import load_config
        hermes_cfg = load_config()
        neural_cfg = hermes_cfg.get("memory", {}).get("neural", {}) or {}
        if isinstance(neural_cfg, dict):
            config.update({k: v for k, v in neural_cfg.items() if v is not None})
    except Exception:
        pass

    # Expand $HERMES_HOME and ~ in db_path
    db_path = config.get("db_path", DEFAULT_DB_PATH)
    if isinstance(db_path, str):
        try:
            from hermes_constants import get_hermes_home
            db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))
            db_path = db_path.replace("${HERMES_HOME}", str(get_hermes_home()))
        except Exception:
            pass
        db_path = os.path.expanduser(db_path)
        config["db_path"] = db_path

    return config
