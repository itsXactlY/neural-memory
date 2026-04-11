"""Neural Memory plugin — MemoryProvider interface.

Local semantic memory with knowledge graph, spreading activation, and
auto-connections. Runs entirely offline — no API keys required.

Stores memories in a local SQLite database with vector embeddings for
semantic recall. Supports multiple embedding backends (hash, tfidf,
sentence-transformers) with auto-detection.

Config via environment variables:
  NEURAL_MEMORY_DB_PATH   — SQLite database path (default: ~/.neural_memory/memory.db)
  NEURAL_EMBEDDING_BACKEND — Embedding backend: auto|hash|tfidf|sentence-transformers

Or via config.yaml:
  memory:
    provider: neural
    neural:
      db_path: ~/.neural_memory/memory.db
      embedding_backend: auto
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure adapter Python modules are importable
# ---------------------------------------------------------------------------

_ADAPTER_PY = Path(__file__).parent
if str(_ADAPTER_PY) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_PY))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = str(Path.home() / ".neural_memory" / "memory.db")
_DEFAULT_EMBEDDING = "auto"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load neural memory config from config.yaml, with env var overrides."""
    config = {
        "db_path": os.environ.get("NEURAL_MEMORY_DB_PATH", _DEFAULT_DB_PATH),
        "embedding_backend": os.environ.get("NEURAL_EMBEDDING_BACKEND", _DEFAULT_EMBEDDING),
    }
    try:
        from hermes_cli.config import load_config
        hermes_cfg = load_config()
        neural_cfg = hermes_cfg.get("memory", {}).get("neural", {}) or {}
        if isinstance(neural_cfg, dict):
            for k, v in neural_cfg.items():
                if v is not None and v != "":
                    config[k] = v
    except Exception:
        pass
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

NEURAL_REMEMBER_SCHEMA = {
    "name": "neural_remember",
    "description": (
        "Store a memory in the neural memory system. "
        "Memories are embedded and auto-connected to similar memories. "
        "Use this for facts, user preferences, decisions, and important context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store.",
            },
            "label": {
                "type": "string",
                "description": "Short label for the memory (optional).",
            },
        },
        "required": ["content"],
    },
}

NEURAL_RECALL_SCHEMA = {
    "name": "neural_recall",
    "description": (
        "Search neural memory using semantic similarity. "
        "Returns memories ranked by relevance with connection info."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
}

NEURAL_THINK_SCHEMA = {
    "name": "neural_think",
    "description": (
        "Spreading activation from a memory — explore connected ideas. "
        "Returns memories activated by traversing the knowledge graph."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "Starting memory ID.",
            },
            "depth": {
                "type": "integer",
                "description": "Activation depth (default: 3).",
            },
        },
        "required": ["memory_id"],
    },
}

NEURAL_GRAPH_SCHEMA = {
    "name": "neural_graph",
    "description": (
        "Get knowledge graph statistics and top connections."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

NEURAL_DREAM_SCHEMA = {
    "name": "neural_dream",
    "description": (
        "Force an immediate dream cycle — autonomous memory consolidation. "
        "Runs NREM (replay & strengthen), REM (explore bridges), and "
        "Insight (community detection) phases. Returns stats from all phases."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "description": "Run only a specific phase (nrem, rem, insight) or all.",
                "enum": ["all", "nrem", "rem", "insight"],
            },
        },
        "required": [],
    },
}

NEURAL_DREAM_STATS_SCHEMA = {
    "name": "neural_dream_stats",
    "description": (
        "Get dream engine statistics — past sessions, connections "
        "strengthened/pruned, bridges found, insights created."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [
    NEURAL_REMEMBER_SCHEMA,
    NEURAL_RECALL_SCHEMA,
    NEURAL_THINK_SCHEMA,
    NEURAL_GRAPH_SCHEMA,
    NEURAL_DREAM_SCHEMA,
    NEURAL_DREAM_STATS_SCHEMA,
]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class NeuralMemoryProvider(MemoryProvider):
    """Neural memory with semantic search, knowledge graph, and spreading activation."""

    # Patterns that indicate meta-reflection, not real content
    _GARBAGE_PATTERNS = (
        "review the conversation above",
        "based on the system prompt",
        "let me review what we know",
        "from the injected context",
        "i can see that",
        "mentioned in the memory section",
        "has invoked the",
        "skill, indicating",
        "let me check my memory",
        "as mentioned in my",
        "according to my memory",
        "i recall from",
    )

    def __init__(self) -> None:
        self._memory = None  # NeuralMemory instance
        self._config: Optional[dict] = None
        self._session_id: str = ""
        self._lock = threading.Lock()
        self._turn_count = 0
        self._prefetch_result: Optional[str] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._initial_context: str = ""
        self._dream_engine = None  # DreamEngine instance

    @property
    def name(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if neural memory dependencies are installed."""
        try:
            import importlib.util
            spec = importlib.util.find_spec("memory_client")
            if spec is None:
                spec = importlib.util.find_spec("embed_provider")
            return spec is not None
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize neural memory for a session."""
        try:
            self._config = _load_config()
            self._session_id = session_id

            from memory_client import NeuralMemory
            self._memory = NeuralMemory(
                db_path=self._config["db_path"],
                embedding_backend=self._config["embedding_backend"],
                use_cpp=True,
            )

            # Start Dream Engine (background consolidation)
            self._start_dream_engine()

            # Eager prefetch: load recent/important context for the first turn
            # so the agent has historical context immediately, not just after turn 1.
            # NOTE: must happen AFTER dream engine start to avoid concurrent DB access
            self._initial_context = self._load_initial_context()

            logger.info(
                "Neural memory initialized: db=%s, backend=%s, dream=%s",
                self._config["db_path"],
                self._config["embedding_backend"],
                "on" if self._dream_engine else "off",
            )
        except ImportError as e:
            logger.warning("Neural memory dependencies not available: %s", e)
            self._memory = None
        except Exception as e:
            logger.warning("Neural memory init failed: %s", e)
            self._memory = None

    def _load_initial_context(self) -> str:
        """Load important context for immediate availability on first turn.

        Queries for:
        1. Recent session summaries
        2. Recent memories (last turns) — direct DB query
        3. High-connection memories (graph hubs = important topics)
        """
        if not self._memory:
            logger.debug("Initial context: no memory instance")
            return ""
        try:
            parts = []
            seen_contents = set()

            # 1. Recent session summaries (exact match)
            summaries = self._memory.recall("session topics recent activity", k=3)
            if summaries:
                for s in summaries:
                    content = s.get("content", "")
                    if "Session topics" in content or "session-summary" in str(s.get("label", "")):
                        if content[:100] not in seen_contents:
                            parts.append(content[:200])
                            seen_contents.add(content[:100])

            # 2. Recent memories — just grab last 5 from DB regardless of query
            try:
                import sqlite3
                db_path = self._config.get("db_path", "")
                if db_path:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT content, label FROM memories "
                        "WHERE label NOT LIKE 'memory-%' "
                        "ORDER BY created_at DESC LIMIT 8"
                    ).fetchall()
                    for row in rows:
                        content = row["content"] or ""
                        # Skip meta-garbage and very short content
                        if len(content) < 30:
                            continue
                        c100 = content[:100]
                        if c100 not in seen_contents and not self._is_garbage(content):
                            parts.append(content[:200])
                            seen_contents.add(c100)
                    conn.close()
            except Exception:
                pass

            # 3. High-connection memories (graph hubs = important topics)
            try:
                graph = self._memory.graph()
                if graph and graph.get("top_connections"):
                    for conn in graph["top_connections"][:3]:
                        weight = conn.get("weight", 0)
                        if weight > 0.5:
                            mid = conn.get("source_id") or conn.get("from_id")
                            if mid:
                                mems = self._memory.recall("", k=50)
                                for m in mems:
                                    if m.get("id") == mid:
                                        content = m.get("content", "")[:200]
                                        if content[:100] not in seen_contents:
                                            parts.append(content)
                                        break
            except Exception:
                pass

            if not parts:
                return ""

            logger.info("Neural initial context: %d items loaded", len(parts))
            return "\n".join(f"- {p}" for p in parts[:10])
        except Exception as e:
            logger.debug("Neural initial context load failed: %s", e)
            return ""

    def _start_dream_engine(self) -> None:
        """Start the dream engine — autonomous background consolidation."""
        try:
            from dream_engine import DreamEngine

            dream_cfg = self._config.get("dream", {})
            if dream_cfg.get("enabled", True) is False:
                logger.info("Dream engine disabled by config")
                return

            idle = dream_cfg.get("idle_threshold", 300)
            threshold = dream_cfg.get("memory_threshold", 50)

            # Check for MSSQL config
            mssql_cfg = dream_cfg.get("mssql", None)
            if mssql_cfg:
                try:
                    self._dream_engine = DreamEngine.mssql(
                        mssql_cfg, self._memory,
                        idle_threshold=idle, memory_threshold=threshold,
                    )
                    logger.info("Dream engine: MSSQL backend")
                except Exception as e:
                    logger.warning("MSSQL dream backend failed, falling back to SQLite: %s", e)
                    mssql_cfg = None

            if not mssql_cfg:
                self._dream_engine = DreamEngine.sqlite(
                    self._config["db_path"], self._memory,
                    idle_threshold=idle, memory_threshold=threshold,
                )
                logger.info("Dream engine: SQLite backend")

            self._dream_engine.start()

        except Exception as e:
            logger.warning("Dream engine failed to start: %s", e)
            self._dream_engine = None

    def system_prompt_block(self) -> str:
        if not self._memory:
            return ""
        try:
            stats = self._memory.stats()
            total = stats.get("memories", 0)
            connections = stats.get("connections", 0)
        except Exception:
            total = 0
            connections = 0

        # Dream stats
        dream_info = ""
        if self._dream_engine:
            try:
                ds = self._dream_engine.get_stats()
                cycles = ds.get("dream_cycles", 0)
                insights = ds.get("total_insights", 0)
                if cycles > 0 or insights > 0:
                    dream_info = f", {cycles} dream cycles, {insights} insights"
            except Exception:
                pass

        # Stack info
        stack_parts = []
        if self._memory and self._memory._cpp:
            stack_parts.append("C++ SIMD")
        if self._memory and self._memory._cosine_sim_fast:
            stack_parts.append("Cython")
        stack_info = f" ({', '.join(stack_parts)})" if stack_parts else ""

        header = (
            f"# Neural Memory{stack_info}\n"
            f"Active. {total} memories, {connections} connections{dream_info}.\n"
            f"Use neural_remember to store new memories.\n"
            f"Use neural_recall to search semantically.\n"
            f"Use neural_think to explore connected ideas.\n"
            f"Use neural_dream to force memory consolidation."
        )

        if total == 0:
            return (
                "# Neural Memory\n"
                "Active. Empty memory store — proactively store facts using neural_remember.\n"
                "Use neural_recall to search memories semantically.\n"
                "Use neural_think to explore connected ideas via spreading activation."
            )

        # Include initial context so agent has historical data from turn 0
        if hasattr(self, "_initial_context") and self._initial_context:
            return (
                f"{header}\n\n"
                f"## Recent Memory Context\n"
                f"(Loaded from previous sessions — use this as background context)\n"
                f"{self._initial_context}"
            )

        return header

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched recall from background thread.

        On the first call (no background result yet), returns initial context
        loaded during initialize() so the agent has historical data immediately.
        """
        if not self._memory or not query:
            return ""
        with self._lock:
            result = self._prefetch_result
            self._prefetch_result = ""

        # First turn: use initial context if no background result yet
        # DON'T consume — system_prompt_block also needs it
        if not result and hasattr(self, "_initial_context") and self._initial_context:
            return f"## Neural Memory Context (recent history)\n{self._initial_context}"

        if not result:
            return ""
        return f"## Neural Memory Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background recall for the next turn."""
        if not self._memory or not query:
            return
        limit = self._config.get("prefetch_limit", 5) if self._config else 5

        def _run():
            try:
                results = self._memory.recall(query, k=limit)
                if not results:
                    return
                lines = []
                for r in results:
                    sim = r.get("similarity", 0)
                    content = r.get("content", "")
                    lines.append(f"- [{sim:.2f}] {content[:200]}")
                with self._lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("Neural prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()

    def _is_garbage(self, text: str) -> bool:
        """Check if text is meta-reflection garbage, not real content."""
        if not text or len(text.strip()) < 20:
            return True
        lower = text.lower().strip()
        for pattern in self._GARBAGE_PATTERNS:
            if pattern in lower:
                return True
        return False

    def _extract_facts(self, user_content: str, assistant_content: str) -> Optional[str]:
        """Extract meaningful facts from a turn, skip garbage."""
        if self._is_garbage(user_content) and self._is_garbage(assistant_content):
            return None
        if user_content.startswith("[SYSTEM:") or user_content.startswith("SYSTEM:"):
            return None
        if self._is_garbage(assistant_content) and len(assistant_content) < 100:
            return None

        user_clean = user_content[:300].strip()
        assist_clean = assistant_content[:500].strip()

        if len(user_clean) < 5:
            return None

        return f"User: {user_clean}\nAssistant: {assist_clean}"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store the turn as an episodic memory. Skips meta-garbage."""
        # Reset dream idle timer on activity
        if self._dream_engine:
            self._dream_engine.touch()
        if not self._memory:
            return

        combined = self._extract_facts(user_content, assistant_content)
        if not combined:
            return

        # Deduplicate: check if very similar content exists
        try:
            existing = self._memory.recall(combined[:100], k=1)
            if existing and existing[0].get("similarity", 0) > 0.95:
                return
        except Exception:
            pass

        self._turn_count += 1
        try:
            self._memory.remember(combined, label=f"turn-{self._turn_count}")
        except Exception as e:
            logger.debug("Neural sync_turn failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract and save important context before compression discards messages.

        Scans user/assistant pairs that are about to be compressed away and
        stores meaningful exchanges as memories so they survive the compression.
        Returns a summary string for the compressor to preserve in its summary.
        """
        if not self._memory or not messages:
            return ""

        extracted = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            user_content = msg.get("content", "")
            if not isinstance(user_content, str) or not user_content.strip():
                continue

            # Find the next assistant response
            assistant_content = ""
            for j in range(i + 1, min(i + 3, len(messages))):
                if messages[j].get("role") == "assistant":
                    assistant_content = messages[j].get("content", "") or ""
                    break

            if not assistant_content:
                continue

            combined = self._extract_facts(user_content, assistant_content)
            if not combined:
                continue

            # Deduplicate against existing memories
            try:
                existing = self._memory.recall(combined[:100], k=1)
                if existing and existing[0].get("similarity", 0) > 0.95:
                    continue
            except Exception:
                pass

            try:
                self._memory.remember(combined, label="pre-compress")
                extracted.append(user_content[:150])
            except Exception as e:
                logger.debug("Neural on_pre_compress remember failed: %s", e)

        if not extracted:
            return ""

        summary = "Key context preserved before compression:\n" + "\n".join(
            f"- {t}" for t in extracted[:20]
        )
        logger.info(
            "Neural memory: saved %d exchanges before compression", len(extracted)
        )
        return summary

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Store a session summary on session end."""
        if not self._memory or not messages:
            return
        try:
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                summary_parts = []
                for m in user_msgs[-10:]:
                    content = m.get("content", "")
                    if isinstance(content, str):
                        summary_parts.append(content[:200])
                if summary_parts:
                    summary = "Session topics: " + " | ".join(summary_parts)
                    self._memory.remember(summary, label="session-summary")
                    logger.info("Neural memory: stored session summary")
        except Exception as e:
            logger.debug("Neural on_session_end failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to neural memory."""
        if action == "add" and self._memory and content:
            if self._is_garbage(content):
                return
            try:
                label = f"memory-{target}" if target else "memory"
                self._memory.remember(content, label=label)
            except Exception as e:
                logger.debug("Neural memory_write mirror failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._memory:
            return tool_error("Neural memory not initialized")
        if tool_name == "neural_remember":
            return self._handle_remember(args)
        elif tool_name == "neural_recall":
            return self._handle_recall(args)
        elif tool_name == "neural_think":
            return self._handle_think(args)
        elif tool_name == "neural_graph":
            return self._handle_graph(args)
        elif tool_name == "neural_dream":
            return self._handle_dream(args)
        elif tool_name == "neural_dream_stats":
            return self._handle_dream_stats(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._dream_engine:
            try:
                self._dream_engine.stop()
            except Exception:
                pass
            self._dream_engine = None
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                pass
            self._memory = None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "Path to SQLite database file",
                "required": False,
                "default": _DEFAULT_DB_PATH,
            },
            {
                "key": "embedding_backend",
                "description": "Embedding backend (auto, hash, tfidf, sentence-transformers)",
                "required": False,
                "default": "auto",
                "choices": ["auto", "hash", "tfidf", "sentence-transformers"],
            },
            {
                "key": "dream.enabled",
                "description": "Enable dream engine (background memory consolidation)",
                "required": False,
                "default": True,
            },
            {
                "key": "dream.idle_threshold",
                "description": "Seconds of idle time before dream cycle (default: 300)",
                "required": False,
                "default": 300,
            },
            {
                "key": "dream.memory_threshold",
                "description": "Dream after this many new memories (default: 50)",
                "required": False,
                "default": 50,
            },
            {
                "key": "dream.mssql",
                "description": "MSSQL config dict for dream backend (server, database, username, password)",
                "required": False,
                "default": None,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write neural config to config.yaml under memory.neural."""
        try:
            from hermes_cli.config import load_config, save_config
            config = load_config()
            neural_cfg = config.setdefault("memory", {}).setdefault("neural", {})
            for k, v in values.items():
                if v is not None and v != "":
                    neural_cfg[k] = v
            save_config(config)
        except Exception as e:
            logger.warning("Failed to save neural config: %s", e)

    # -- Tool handlers -------------------------------------------------------

    def _handle_remember(self, args: dict) -> str:
        try:
            content = args["content"]
            label = args.get("label", "")
            mem_id = self._memory.remember(content, label=label)
            return json.dumps({"id": mem_id, "status": "stored"})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_recall(self, args: dict) -> str:
        try:
            query = args["query"]
            limit = int(args.get("limit", 5))
            results = self._memory.recall(query, k=limit)
            # Strip embedding vectors and connections — only send useful fields to LLM
            clean = []
            for r in results:
                clean.append({
                    "id": r.get("id"),
                    "label": r.get("label", ""),
                    "content": r.get("content", "")[:500],
                    "similarity": round(r.get("similarity", 0), 3),
                })
            return json.dumps({"results": clean, "count": len(clean)})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_think(self, args: dict) -> str:
        try:
            memory_id = int(args["memory_id"])
            depth = int(args.get("depth", 3))
            results = self._memory.think(memory_id, depth=depth)
            # Strip internal fields
            clean = []
            for r in results:
                clean.append({
                    "id": r.get("id"),
                    "label": r.get("label", ""),
                    "content": (r.get("content", "") or "")[:300],
                    "activation": round(r.get("activation", 0), 3),
                })
            return json.dumps({"results": clean, "count": len(clean)})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_graph(self, args: dict) -> str:
        try:
            graph = self._memory.graph()
            stats = self._memory.stats()
            return json.dumps({"graph": graph, "stats": stats})
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_dream(self, args: dict) -> str:
        """Force a dream cycle (or specific phase)."""
        if not self._dream_engine:
            return tool_error("Dream engine not running")
        try:
            phase = args.get("phase", "all")
            if phase == "all":
                result = self._dream_engine.dream_now()
            else:
                # Run specific phase
                method_map = {
                    "nrem": self._dream_engine._phase_nrem,
                    "rem": self._dream_engine._phase_rem,
                    "insight": self._dream_engine._phase_insights,
                }
                method = method_map.get(phase)
                if not method:
                    return tool_error(f"Unknown phase: {phase}")
                result = method()
            return json.dumps({"status": "complete", "phase": phase, "stats": result})
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_dream_stats(self, args: dict) -> str:
        """Return dream engine statistics."""
        if not self._dream_engine:
            return tool_error("Dream engine not running")
        try:
            stats = self._dream_engine.get_stats()
            return json.dumps(stats)
        except Exception as exc:
            return tool_error(str(exc))


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the neural memory provider with the plugin system."""
    provider = NeuralMemoryProvider()
    ctx.register_memory_provider(provider)
