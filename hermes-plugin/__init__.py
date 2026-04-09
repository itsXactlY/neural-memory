from __future__ import annotations

"""Neural Memory plugin - MemoryProvider for Neural Memory Adapter.

Provides semantic memory storage with embedding-based recall, knowledge graph
connections, and spreading activation via the neural-memory-adapter Python
client (memory_client.py + embed_provider.py).

Config (in ~/.hermes/config.yaml):
  memory:
    provider: neural
    neural:
      db_path: ~/.neural_memory/hermes.db
      embedding_backend: auto
      consolidation_interval: 300
      max_episodic: 50000
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

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
                "description": "Short label for the memory (optional, auto-generated from content if omitted).",
            },
        },
        "required": ["content"],
    },
}

NEURAL_RECALL_SCHEMA = {
    "name": "neural_recall",
    "description": (
        "Search neural memory using semantic similarity. "
        "Returns memories ranked by relevance with connection info. "
        "Use this to recall past conversations, facts, or user preferences."
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
        "Returns memories activated by traversing the knowledge graph from a starting point. "
        "Use to find related context that isn't directly similar."
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
        "Get knowledge graph statistics and top connections. "
        "Use to understand the structure of stored memories."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [
    NEURAL_REMEMBER_SCHEMA,
    NEURAL_RECALL_SCHEMA,
    NEURAL_THINK_SCHEMA,
    NEURAL_GRAPH_SCHEMA,
]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class NeuralMemoryProvider(MemoryProvider):
    """Neural memory with semantic search, knowledge graph, and spreading activation."""

    def __init__(self):
        self._memory: Optional[Any] = None  # NeuralMemory instance
        self._config: Optional[dict] = None
        self._session_id: str = ""
        self._lock = threading.Lock()
        self._prefetch_result: str = ""
        self._prefetch_thread: Optional[threading.Thread] = None
        self._consolidation_thread: Optional[threading.Thread] = None
        self._consolidation_running = False
        self._turn_count = 0

    @property
    def name(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if neural memory dependencies are installed."""
        try:
            from plugins.memory.neural.config import get_config
            cfg = get_config()
            # Try importing the memory client
            import importlib.util
            spec = importlib.util.find_spec("memory_client")
            if spec is None:
                # Try from the project directory
                import sys
                from pathlib import Path
                project_py = str(Path.home() / "projects" / "neural-memory-adapter" / "python")
                if project_py not in sys.path:
                    sys.path.insert(0, project_py)
                spec = importlib.util.find_spec("memory_client")
            return spec is not None
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize neural memory for a session."""
        try:
            from plugins.memory.neural.config import get_config
            self._config = get_config()
            self._session_id = session_id

            # Ensure the python directory is on sys.path
            import sys
            from pathlib import Path
            project_py = str(Path.home() / "projects" / "neural-memory-adapter" / "python")
            if project_py not in sys.path:
                sys.path.insert(0, project_py)

            from memory_client import NeuralMemory

            self._memory = NeuralMemory(
                db_path=self._config["db_path"],
                embedding_backend=self._config["embedding_backend"],
            )

            # Start background consolidation thread
            self._start_consolidation_thread()

            logger.info(
                "Neural memory initialized: db=%s, backend=%s",
                self._config["db_path"],
                self._config["embedding_backend"],
            )
        except ImportError as e:
            logger.warning("Neural memory dependencies not available: %s", e)
            self._memory = None
        except Exception as e:
            logger.warning("Neural memory init failed: %s", e)
            self._memory = None

    def _start_consolidation_thread(self) -> None:
        """Start background consolidation thread."""
        if not self._config:
            return
        interval = self._config.get("consolidation_interval", 300)
        if interval <= 0:
            return

        self._consolidation_running = True

        def _consolidate_loop():
            while self._consolidation_running:
                time.sleep(interval)
                if not self._consolidation_running:
                    break
                try:
                    self._run_consolidation()
                except Exception as e:
                    logger.debug("Consolidation error: %s", e)

        self._consolidation_thread = threading.Thread(
            target=_consolidate_loop, daemon=True, name="neural-consolidation"
        )
        self._consolidation_thread.start()

    def _run_consolidation(self) -> None:
        """Run consolidation: prune low-salience episodic memories."""
        if not self._memory or not self._config:
            return
        max_episodic = self._config.get("max_episodic", 50000)
        try:
            stats = self._memory.stats()
            total = stats.get("memories", 0)
            if total > max_episodic:
                # In a full implementation, we'd prune low-salience memories
                # For now, log a warning
                logger.info(
                    "Neural memory: %d memories (max %d) — consolidation active",
                    total, max_episodic,
                )
        except Exception as e:
            logger.debug("Consolidation check failed: %s", e)

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

        if total == 0:
            return (
                "# Neural Memory\n"
                "Active. Empty memory store — proactively store facts the user would expect "
                "you to remember using neural_remember.\n"
                "Use neural_recall to search memories semantically.\n"
                "Use neural_think to explore connected ideas via spreading activation."
            )
        return (
            f"# Neural Memory\n"
            f"Active. {total} memories, {connections} connections.\n"
            f"Use neural_remember to store new memories.\n"
            f"Use neural_recall to search semantically.\n"
            f"Use neural_think to explore connected ideas."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched recall from background thread."""
        if not self._memory or not query:
            return ""
        with self._lock:
            result = self._prefetch_result
            self._prefetch_result = ""
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

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store the turn as an episodic memory."""
        if not self._memory:
            return
        self._turn_count += 1
        try:
            combined = f"User: {user_content}\nAssistant: {assistant_content[:500]}"
            self._memory.remember(combined, label=f"turn-{self._turn_count}")
        except Exception as e:
            logger.debug("Neural sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "neural_remember":
            return self._handle_remember(args)
        elif tool_name == "neural_recall":
            return self._handle_recall(args)
        elif tool_name == "neural_think":
            return self._handle_think(args)
        elif tool_name == "neural_graph":
            return self._handle_graph(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Consolidate on session end - store conversation summary."""
        if not self._memory or not messages:
            return
        try:
            # Store a session summary
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                summary_parts = []
                for m in user_msgs[-10:]:  # Last 10 user messages
                    content = m.get("content", "")
                    if isinstance(content, str):
                        summary_parts.append(content[:200])
                if summary_parts:
                    summary = "Session topics: " + " | ".join(summary_parts)
                    self._memory.remember(summary, label="session-summary")
                    logger.info("Neural memory: stored session summary")
        except Exception as e:
            logger.debug("Neural on_session_end failed: %e", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to neural memory."""
        if action == "add" and self._memory and content:
            try:
                label = f"memory-{target}" if target else "memory"
                self._memory.remember(content, label=label)
            except Exception as e:
                logger.debug("Neural memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._consolidation_running = False
        if self._consolidation_thread and self._consolidation_thread.is_alive():
            self._consolidation_thread.join(timeout=2.0)
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                pass
            self._memory = None

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
            return json.dumps({"results": results, "count": len(results)})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_think(self, args: dict) -> str:
        try:
            memory_id = int(args["memory_id"])
            depth = int(args.get("depth", 3))
            results = self._memory.think(memory_id, depth=depth)
            return json.dumps({"results": results, "count": len(results)})
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the neural memory provider with the plugin system."""
    provider = NeuralMemoryProvider()
    ctx.register_memory_provider(provider)
    
    
