from __future__ import annotations

"""Mazemaker plugin - MemoryProvider for Mazemaker Adapter.

Provides semantic memory storage with embedding-based recall, knowledge graph
connections, and spreading activation via the mazemaker-adapter Python
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

MAZEMAKER_REMEMBER_SCHEMA = {
    "name": "mazemaker_remember",
    "description": (
        "Store a memory in the mazemaker system. "
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

MAZEMAKER_RECALL_SCHEMA = {
    "name": "mazemaker_recall",
    "description": (
        "Search mazemaker using semantic similarity. "
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

MAZEMAKER_THINK_SCHEMA = {
    "name": "mazemaker_think",
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

MAZEMAKER_GRAPH_SCHEMA = {
    "name": "mazemaker_graph",
    "description": (
        "Get knowledge graph statistics and top connections. "
        "Use to understand the structure of stored memories."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [
    MAZEMAKER_REMEMBER_SCHEMA,
    MAZEMAKER_RECALL_SCHEMA,
    MAZEMAKER_THINK_SCHEMA,
    MAZEMAKER_GRAPH_SCHEMA,
]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class NeuralMemoryProvider(MemoryProvider):
    """Neural memory with semantic search, knowledge graph, and spreading activation."""

    def __init__(self):
        self._memory: Optional[Any] = None  # Mazemaker instance
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
        """Check if mazemaker dependencies are installed."""
        try:
            import sys
            from pathlib import Path

            # Add project python dir to path
            project_py = str(Path.home() / "projects" / "mazemaker-adapter" / "python")
            if project_py not in sys.path:
                sys.path.insert(0, project_py)

            # Add plugin dir to path
            plugin_dir = str(Path(__file__).parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            # Actually try importing
            from mazemaker import Memory
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("neural not available: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize mazemaker for a session."""
        try:
            import sys
            import os
            from pathlib import Path

            # Ensure paths are on sys.path
            project_py = str(Path.home() / "projects" / "mazemaker-adapter" / "python")
            if project_py not in sys.path:
                sys.path.insert(0, project_py)
            plugin_dir = str(Path(__file__).parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            from config import get_config
            self._config = get_config()
            self._session_id = session_id

            # Use Memory class (auto-detects Postgres vs SQLite)
            from mazemaker import Memory
            self._memory = Memory(
                db_path=self._config["db_path"],
                embedding_backend=self._config["embedding_backend"],
            )

            # Start dream engine
            self._start_dream_engine()

            # Start background consolidation thread
            self._start_consolidation_thread()

            backend = self._memory.backend if hasattr(self._memory, 'backend') else 'unknown'
            logger.info(
                "Neural memory initialized: db=%s, backend=%s",
                self._config["db_path"],
                backend,
            )
        except ImportError as e:
            logger.warning("Neural memory dependencies not available: %s", e)
            self._memory = None
        except Exception as e:
            logger.warning("Neural memory init failed: %s", e)
            self._memory = None

    def _start_dream_engine(self) -> None:
        """Start the SQLite-backed dream engine."""
        from pathlib import Path

        try:
            from dream_engine import DreamEngine

            db_path = self._config.get("db_path", str(Path.home() / ".neural_memory" / "memory.db"))
            self._dream = DreamEngine.sqlite(
                db_path,
                neural_memory=self._memory,
                idle_threshold=600,
                memory_threshold=50,
                min_dream_interval=600,
            )
            self._dream.start()
            logger.info("Dream engine started: SQLite")

        except Exception as e:
            logger.warning("Dream engine failed to start: %s", e)
            self._dream = None

    def _start_consolidation_thread(self) -> None:
        """Start background consolidation thread."""
        if not self._config:
            return
        interval = self._config.get("consolidation_interval", 0)
        if interval <= 0:
            return  # Consolidation disabled

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
        max_episodic = self._config.get("max_episodic", 0)
        if max_episodic <= 0:
            return  # Unlimited - no pruning
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
                "# Mazemaker\n"
                "Active. Empty memory store — proactively store facts the user would expect "
                "you to remember using neural_remember.\n"
                "Use neural_recall to search memories semantically.\n"
                "Use neural_think to explore connected ideas via spreading activation."
            )
        return (
            f"# Mazemaker\n"
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
        return f"## Mazemaker Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background recall for the next turn."""
        if not self._memory or not query:
            return
        limit = min(self._config.get("prefetch_limit", 3), 3) if self._config else 3

        def _run():
            try:
                results = self._memory.recall(query, k=limit * 3)  # Over-fetch, then filter
                if not results:
                    return
                lines = []
                for r in results:
                    sim = r.get("similarity", 0)
                    if sim < 0.5:
                        continue  # Skip low-quality matches
                    content = r.get("content", "")
                    # Skip meta/debug/angry content
                    content_lower = content.lower()
                    if any(skip in content_lower for skip in (
                        "mazemaker", "tool_result", "test_suite",
                        "config.yaml", "embedding", "connection string",
                        "scheiße", "dreck", "huren", "verfickt",
                        "system note", "memory-context",
                    )):
                        continue
                    # Skip very short content
                    if len(content) < 30:
                        continue
                    lines.append(f"- [{sim:.2f}] {content[:150]}")
                    if len(lines) >= limit:
                        break
                if lines:
                    with self._lock:
                        self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("Neural prefetch failed: %s", e)
        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()

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
        "mazemaker",
        "mazemaker_recall",
        "mazemaker_remember",
        "does mazemaker work",
        "tool_result",
        "test_suite",
        "config.yaml",
        "sqlite",
        "postgres",
        "pgvector",
        "embedding",
        "connection string",
    )

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
        # Skip if both are garbage
        if self._is_garbage(user_content) and self._is_garbage(assistant_content):
            return None

        # Skip system/tool messages
        if user_content.startswith("[SYSTEM:") or user_content.startswith("SYSTEM:"):
            return None

        # Skip if assistant just says "let me check" without substance
        if self._is_garbage(assistant_content) and len(assistant_content) < 100:
            return None

        # Build clean memory
        user_clean = user_content[:300].strip()
        assist_clean = assistant_content[:500].strip()

        # Only store if user asked something real
        if len(user_clean) < 5:
            return None

        return f"User: {user_clean}\nAssistant: {assist_clean}"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """No-op. Per-turn storage disabled — only session summaries stored at session end.
        
        Old behaviour stored every turn as a separate memory, creating a feedback loop
        where recent conversation turns were recalled as 'context' and re-injected.
        The proper approach: on_session_end stores a single session summary.
        """
        self._turn_count += 1

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mazemaker_remember":
            return self._handle_remember(args)
        elif tool_name == "mazemaker_recall":
            return self._handle_recall(args)
        elif tool_name == "mazemaker_think":
            return self._handle_think(args)
        elif tool_name == "mazemaker_graph":
            return self._handle_graph(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Store a session summary at session end — the ONLY memory write per session."""
        if not self._memory or not messages:
            return
        try:
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                return
            
            summary_parts = []
            for m in user_msgs:
                content = m.get("content", "")
                if not isinstance(content, str):
                    continue
                # Skip noise: tool results, log dumps, very short messages
                if len(content) < 10:
                    continue
                if content.startswith(("[SYSTEM:", "SYSTEM:", "Tool ", "Batches:")):
                    continue
                if any(skip in content.lower() for skip in (
                    "tool_result_storage", "run_agent", "DEBUG", "openai client",
                    "token usage", "completion_tokens", "snapshot_engine",
                )):
                    continue
                # Extract first meaningful line
                first_line = content.split("\n")[0][:150]
                if first_line:
                    summary_parts.append(first_line)
            
            if summary_parts:
                summary = "Session: " + " | ".join(summary_parts[-8:])  # Last 8 meaningful lines
                self._memory.remember(summary, label="session-summary")
                logger.info("Neural memory: stored session summary (%d topics)", len(summary_parts))
        except Exception as e:
            logger.debug("Neural on_session_end failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to mazemaker. Skips garbage."""
        if action == "add" and self._memory and content:
            if self._is_garbage(content):
                return
            try:
                label = f"memory-{target}" if target else "memory"
                self._memory.remember(content, label=label)
            except Exception as e:
                logger.debug("Neural memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        """Clean shutdown."""
        # Stop dream engine
        if hasattr(self, '_dream') and self._dream:
            try:
                self._dream.stop()
            except Exception:
                pass
            self._dream = None
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
            # Touch dream engine (reset idle timer)
            if hasattr(self, '_dream') and self._dream:
                self._dream.touch()
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
    """Register the mazemaker provider with the plugin system."""
    provider = NeuralMemoryProvider()
    ctx.register_memory_provider(provider)
    
    
