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

import hashlib
import json
import logging
import queue
import re
import threading
from pathlib import Path
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
        self._prefetch_result: Optional[str] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        # Monotonic sequence number for prefetch generations. Only the
        # newest prefetch is allowed to publish into _prefetch_result —
        # an in-flight slow one whose generation has been superseded
        # silently drops its result instead of clobbering a fresher one.
        self._prefetch_gen: int = 0
        self._initial_context: str = ""
        self._consolidation_thread: Optional[threading.Thread] = None
        self._consolidation_stop = threading.Event()  # set = stop requested
        self._turn_count = 0
        self._dream = None  # DreamEngine instance
        self._dream_was_running_before_turn = False  # track if dreaming when turn started
        # Sponge mode: immediate background absorption
        self._sponge_queue: Optional[queue.Queue] = None
        self._sponge_worker: Optional[threading.Thread] = None
        self._sponge_running = False

    @property
    def name(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if neural memory dependencies are installed."""
        try:
            import sys
            from pathlib import Path

            # Resolve symlinks so imports work when running from the symlink
            plugin_dir = str(Path(__file__).resolve().parent)
            real_project_dir = str(Path(__file__).resolve().parent.parent.parent / "neural-memory-adapter" / "python")

            for p in (plugin_dir, real_project_dir):
                if p not in sys.path:
                    sys.path.insert(0, p)

            # Actually try importing
            from neural_memory import Memory
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("neural not available: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize neural memory for a session."""
        try:
            import sys
            import os
            from pathlib import Path

            # Ensure real plugin source dir is on sys.path. __file__ may be a
            # symlink under ~/.hermes/plugins/...; Path(__file__).parent alone
            # points at a sparse runtime dir that may not contain config.py or
            # memory_client.py. Resolve to the source-of-truth python/ dir.
            plugin_dir = str(Path(__file__).resolve().parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            from config import get_config
            self._config = get_config()
            self._session_id = session_id

            # Set MSSQL env vars from config.yaml — single source of truth.
            # C++ bridge reads these via std::getenv(), not from config dict.
            mssql_cfg = self._config.get("dream", {}).get("mssql", {})
            env_map = {
                "MSSQL_SERVER": mssql_cfg.get("server", ""),
                "MSSQL_DATABASE": mssql_cfg.get("database", ""),
                "MSSQL_USERNAME": mssql_cfg.get("username", ""),
                "MSSQL_PASSWORD": mssql_cfg.get("password", ""),
                "MSSQL_DRIVER": mssql_cfg.get("driver", "{ODBC Driver 18 for SQL Server}"),
            }
            for key, val in env_map.items():
                if val:
                    os.environ[key] = str(val)

            # Use Memory class (auto-detects MSSQL vs SQLite)
            from neural_memory import Memory
            def _cfg_bool(value, default=False):
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}
            self._memory = Memory(
                db_path=self._config["db_path"],
                embedding_backend=self._config["embedding_backend"],
                retrieval_mode=self._config.get("retrieval_mode", "semantic"),
                retrieval_candidates=int(self._config.get("retrieval_candidates", 64) or 64),
                use_hnsw=self._config.get("use_hnsw", "auto"),
                lazy_graph=_cfg_bool(self._config.get("lazy_graph"), False),
                think_engine=self._config.get("think_engine", "bfs"),
                rerank=_cfg_bool(self._config.get("rerank"), False),
                channel_weights=self._config.get("channel_weights"),
                rrf_k=int(self._config.get("rrf_k", 60) or 60),
                salience_decay_k=float(self._config.get("salience_decay_k", 0.03) or 0.03),
                ppr_alpha=float(self._config.get("ppr_alpha", 0.15) or 0.15),
                ppr_iters=int(self._config.get("ppr_iters", 20) or 20),
                ppr_hops=int(self._config.get("ppr_hops", 2) or 2),
            )

            # Load initial context from memory — DEFERRED to a daemon thread.
            # Calling _load_initial_context() synchronously here was the main
            # contributor to the >3min hermes-startup latency: it fires two
            # recall() calls, each of which lazily loads the embed model and
            # builds the HNSW index on first call. Backgrounding it lets
            # init return immediately; the first turn's system_prompt_block
            # consumes _initial_context if it has landed by then, otherwise
            # it gracefully starts blank.
            self._initial_context = ""
            self._initial_context_thread = threading.Thread(
                target=self._load_initial_context_async,
                daemon=True,
                name="neural-initial-ctx",
            )
            self._initial_context_thread.start()

            # Start dream engine
            self._start_dream_engine()

            # Start background consolidation thread
            self._start_consolidation_thread()

            # Start Sponge worker (immediate message absorption)
            self._start_sponge()

            backend = self._memory.backend if hasattr(self._memory, 'backend') else 'unknown'
            logger.info(
                "Neural memory initialized: db=%s, backend=%s, mssql=%s",
                self._config["db_path"],
                backend,
                self._memory._use_mssql if hasattr(self._memory, '_use_mssql') else False,
            )
        except ImportError as e:
            logger.warning("Neural memory dependencies not available: %s", e)
            self._memory = None
        except Exception as e:
            logger.warning("Neural memory init failed: %s", e)
            self._memory = None

    def update_session_id(self, session_id: str) -> None:
        """Called after session split (e.g. compression) to update the session tag.
        
        Neural Memory uses session_id as the archive_tag for archive_compression().
        After compression creates a new session_id, this ensures subsequent
        archives and prefetches use the correct new tag.
        """
        self._session_id = session_id
        logger.debug("Neural memory session_id updated: %s", session_id)

    def _start_dream_engine(self) -> None:
        """Start dream engine — MSSQL (C++) if available, SQLite fallback.

        Idempotent: if a prior DreamEngine is still running (e.g. caller is
        re-initialising NeuralMemory after a session split or model switch),
        stop it first.  Without this each call leaks a daemon thread plus
        its DB handle and C++ backend state.
        """
        import os
        from pathlib import Path

        if self._dream is not None:
            try:
                self._dream.stop()
            except Exception as e:
                logger.debug("Prior dream engine stop failed: %s", e)
            self._dream = None

        try:
            from dream_engine import DreamEngine

            # Check if MSSQL is configured
            mssql_server = os.environ.get("MSSQL_SERVER", "")
            mssql_password = os.environ.get("MSSQL_PASSWORD", "")

            if mssql_server and mssql_password:
                # MSSQL active → use C++ dream backend
                try:
                    from cpp_dream_backend import CppDreamBackend
                    backend = CppDreamBackend(dim=self._memory.dim if hasattr(self._memory, 'dim') else 1024)
                    self._dream = DreamEngine(
                        backend,
                        neural_memory=self._memory,
                        idle_threshold=600,
                        memory_threshold=50,
                    )
                    self._dream.start()
                    logger.info("Dream engine started: C++ → MSSQL")
                    return
                except Exception as e:
                    logger.warning("C++ dream backend failed, falling back: %s", e)

            # SQLite fallback — use memory.db which has all tables (memories, connections, dream_*)
            db_path = self._config.get("db_path", str(Path.home() / ".neural_memory" / "memory.db"))
            self._dream = DreamEngine.sqlite(
                db_path,
                neural_memory=self._memory,
                idle_threshold=600,
                memory_threshold=50,
            )
            self._dream.start()
            logger.info("Dream engine started: SQLite fallback")

        except Exception as e:
            logger.warning("Dream engine failed to start: %s", e)
            self._dream = None

    def _start_consolidation_thread(self) -> None:
        """Start background consolidation thread.

        Idempotent: signals any existing loop to stop and joins it before
        spawning a replacement, so re-initialisation can't leak threads.
        """
        if not self._config:
            return
        interval = self._config.get("consolidation_interval", 0)
        if interval <= 0:
            return  # Consolidation disabled

        prior = getattr(self, "_consolidation_thread", None)
        if prior is not None and prior.is_alive():
            self._consolidation_stop.set()
            prior.join(timeout=3.0)

        self._consolidation_stop.clear()

        def _consolidate_loop():
            while not self._consolidation_stop.is_set():
                if self._consolidation_stop.wait(timeout=interval):
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

    def _load_initial_context(self) -> str:
        """Query for session summaries, recent memories, graph hubs."""
        if not self._memory:
            return ""
        try:
            parts = []
            # Recent session summaries
            summaries = self._memory.recall("session topics recent activity", k=3)
            for s in summaries:
                if "Session topics" in s.get("content", ""):
                    parts.append(s["content"][:200])
            # Recent memories
            recent = self._memory.recall("recent conversation context", k=5)
            for r in recent:
                if r.get("similarity", 0) > 0.2:
                    parts.append(r["content"][:200])
            return "\n".join(f"- {p}" for p in parts[:10]) if parts else ""
        except Exception as e:
            logger.debug("Failed to load initial context: %s", e)
            return ""

    def _load_initial_context_async(self) -> None:
        """Background entrypoint — populates self._initial_context off the
        critical path. Used by initialize() so the provider returns fast
        even when the first recall has to lazy-load the embed model and
        build the HNSW index. Errors are swallowed silently — the
        consumer (system_prompt_block / prefetch fallback) treats an
        unset initial context as 'no recent activity'.
        """
        try:
            ctx = self._load_initial_context()
        except Exception:
            ctx = ""
        with self._lock:
            self._initial_context = ctx or ""
        logger.debug("Neural initial context loaded (%d chars)", len(self._initial_context))

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
            header = (
                "# Neural Memory\n"
                "Active. Empty memory store — proactively store facts the user would expect "
                "you to remember using neural_remember.\n"
                "Use neural_recall to search memories semantically.\n"
                "Use neural_think to explore connected ideas via spreading activation."
            )
        else:
            header = (
                f"# Neural Memory\n"
                f"Active. {total} memories, {connections} connections.\n"
                f"Use neural_remember to store new memories.\n"
                f"Use neural_recall to search semantically.\n"
                f"Use neural_think to explore connected ideas."
            )

        if self._initial_context:
            return f"{header}\n\n## Recent Memory Context\n{self._initial_context}"
        return header

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched recall from background thread."""
        if not self._memory or not query:
            return ""
        with self._lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            # On first call, return initial context if available
            if self._initial_context:
                return f"## Neural Memory Context (recent history)\n{self._initial_context}"
            return ""
        return f"## Neural Memory Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background recall for the next turn.

        If an earlier prefetch is still in flight, this call bumps the
        generation counter so the older thread's result is dropped on
        completion (see `_prefetch_gen` check below). That prevents a slow
        prefetch from clobbering a fresher one with stale data, and makes
        the most-recent queue_prefetch the source of truth.
        """
        if not self._memory or not query:
            return
        limit = min(self._config.get("prefetch_limit", 3), 3) if self._config else 3

        with self._lock:
            self._prefetch_gen += 1
            my_gen = self._prefetch_gen

        def _run():
            try:
                results = self._memory.recall(query, k=limit * 2)  # Over-fetch, then filter
                if not results:
                    return
                lines = []
                for r in results:
                    sim = r.get("similarity", 0)
                    if sim < 0.5:
                        continue  # Skip low-quality matches
                    content = r.get("content", "")
                    # Skip meta/debug content
                    content_lower = content.lower()
                    if any(skip in content_lower for skip in (
                        "neural memory", "tool_result", "test_suite", "mssql",
                        "config.yaml", "odbc", "embedding", "connection string",
                        "archive:session",
                    )):
                        continue
                    lines.append(f"- [{sim:.2f}] {content[:150]}")
                    if len(lines) >= limit:
                        break
                if lines:
                    with self._lock:
                        # Drop our result if a newer prefetch has been queued
                        # in the meantime — its result is the one the next
                        # turn should see, not ours.
                        if my_gen == self._prefetch_gen:
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
        "neural memory",
        "neural_recall",
        "neural_remember",
        "does neural memory work",
        "tool_result",
        "test_suite",
        "config.yaml",
        "mssql",
        "sqlite",
        "embedding",
        "connection string",
        "odbc",
        # Compaction artifacts — these indicate stale/loop content
        "[SUPERSEDED]",
        "[UPDATED TO]",
        "[CONTEXT COMPACTION",
        "[SYSTEM: If you have a meaningful status report",
        "context compaction — reference only",
        "the latest user message that appears after this summary",
    )

    # Patterns that indicate the text is NOT a useful memory (banner/log/config)
    _NOISE_LABELS = frozenset({
        "pre-compress",
    })
    # Label prefixes that are auto-generated noise (cron reports, gateway msgs)
    _NOISE_LABEL_PREFIXES = ("msg:",)

    def _is_noise_label(self, label: str) -> bool:
        """Check if a label indicates auto-generated noise."""
        if label in self._NOISE_LABELS:
            return True
        return any(label.startswith(p) for p in self._NOISE_LABEL_PREFIXES)

    def _start_sponge(self) -> None:
        """Start the sponge worker thread for immediate message absorption."""
        if self._sponge_running:
            return
        self._sponge_queue = queue.Queue(maxsize=100)
        self._sponge_running = True
        self._sponge_worker = threading.Thread(
            target=self._sponge_loop, daemon=True, name="neural-sponge"
        )
        self._sponge_worker.start()
        logger.debug("Neural sponge worker started")

    def _stop_sponge(self) -> None:
        """Stop the sponge worker thread."""
        self._sponge_running = False
        if self._sponge_queue:
            try:
                self._sponge_queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass
        if self._sponge_worker and self._sponge_worker.is_alive():
            self._sponge_worker.join(timeout=3)
        self._sponge_worker = None
        self._sponge_queue = None

    def _sponge_loop(self) -> None:
        """Background worker: drain queue and store memories."""
        while self._sponge_running:
            try:
                item = self._sponge_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:  # sentinel
                break
            role, content = item
            try:
                self._do_absorb(role, content)
            except Exception as e:
                logger.debug("Sponge absorb failed: %s", e)

    def _do_absorb(self, role: str, content: str) -> None:
        """Actually store a message as a memory (called from sponge worker)."""
        if not self._memory:
            return
        if self._is_garbage(content):
            return

        # Extract meaningful content based on role
        if role == "user":
            label = f"user-msg"
            memory_text = f"Q: {content[:500]}"
        else:
            # For assistant: check if it's a non-answer
            assist_lower = content.lower()[:300]
            _non_answers = (
                "i don't have", "i don't know", "i can't find",
                "no specific memory", "memory is incomplete",
                "i don't recall", "beyond what's in my notes",
                "can you remind me", "nothing specific about",
            )
            if any(n in assist_lower for n in _non_answers):
                return  # Don't store non-answers
            label = f"asst-msg"
            memory_text = f"A: {content[:500]}"

        # Deduplicate: skip if very similar content already exists
        try:
            existing = self._memory.recall(memory_text[:100], k=1)
            if existing and existing[0].get("similarity", 0) > 0.95:
                # Extra check: verify actual content overlap, not just embedding similarity
                existing_content = (existing[0].get("content", "") or "")[:100]
                if existing_content and existing_content == memory_text[:len(existing_content)]:
                    return  # Exact duplicate
        except Exception:
            pass

        try:
            self._memory.remember(memory_text, label=label)
        except Exception as e:
            logger.debug("Sponge remember failed: %s", e)

    def absorb_message(self, role: str, content: str) -> None:
        """Queue a message for immediate background absorption.

        Call this for EVERY message as it arrives — user messages before
        processing, assistant messages after generation. Non-blocking.

        Args:
            role: 'user' or 'assistant'
            content: The message text
        """
        if not self._sponge_running or not self._memory:
            return
        if not content or len(content.strip()) < 10:
            return
        try:
            self._sponge_queue.put_nowait((role, content))
        except queue.Full:
            logger.debug("Sponge queue full, dropping message")


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

        # Skip boilerplate responses
        _boilerplate = [
            "trallala", "got it", "ok", "sure", "thanks", "thank you",
            "i understand", "i see", "alright", "okay", "right",
        ]
        assist_lower = assist_clean.lower()
        is_boilerplate = any(b in assist_lower for b in _boilerplate) and len(assist_clean) < 200
        if is_boilerplate or not assist_clean:
            return f"Topic: {user_clean[:300]}"
        return f"Topic: {user_clean[:200]}\nResult: {assist_clean[:300]}"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """No-op. Per-turn storage disabled — only session summaries stored at session end.
        
        Old behaviour stored every turn as a separate memory, creating a feedback loop
        where recent conversation turns were recalled as 'context' and re-injected.
        The proper approach: on_session_end stores a single session summary.
        """
        self._turn_count += 1

    def post_llm_call(self, session_id: str, user_message: str, assistant_response: str,
                      conversation_history: list, model: str, platform: str, **kwargs) -> None:
        """After every LLM answer: resume dream engine and optionally archive raw turns.

        Durable memory extraction happens in on_session_end(). Raw per-turn writes are
        intentionally opt-in because raw chat/tool dumps poison recall and can expose
        private conversation text. Set memory.neural.store_raw_turns/archive_raw_turns
        to true only for forensic debugging.
        """
        # 1. Dream engine resume (existing behaviour)
        if self._dream is not None and self._dream_was_running_before_turn:
            self._dream.start()
            self._dream_was_running_before_turn = False

        if not self._memory:
            return

        def _cfg_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

        store_raw = _cfg_bool((self._config or {}).get("store_raw_turns"), False)
        archive_raw = _cfg_bool((self._config or {}).get("archive_raw_turns"), False)

        if store_raw:
            for role, content, label in (
                ("user", user_message, "turn:user"),
                ("assistant", assistant_response, "turn:assistant"),
            ):
                if content and len(str(content).strip()) >= 3:
                    try:
                        self._memory.remember(str(content), label=label)
                    except Exception as e:
                        logger.debug("Raw %s message store failed: %s", role, e)

        if archive_raw and conversation_history:
            try:
                session_tag = f"session-{session_id[:8]}" if session_id else "session-unknown"
                self._memory.archive_compression(
                    turns=conversation_history[-6:],
                    session_tag=session_tag,
                )
            except Exception as e:
                logger.debug("Turn archive failed: %s", e)

    def _on_pre_llm_call(self, session_id: str, user_message: str, **kwargs) -> None:
        """Internal: activity signal from pre_llm_call hook.
        
        Registered as a plugin hook (pre_llm_call) to get notified on every turn.
        This is the PRIMARY activity signal — fires once per turn, before any tool
        calls or LLM processing.
        
        Pause/resume pattern:
        - pre_llm_call: record if dreaming, then pause, touch idle timer
        - post_llm_call: resume if it was running and still idle
        """
        if self._dream is None:
            return
        
        # Record whether the dream engine was running when this turn started
        self._dream_was_running_before_turn = (
            hasattr(self._dream, '_thread') 
            and self._dream._thread is not None 
            and self._dream._thread.is_alive()
        )
        
        # Stop the dream engine for the duration of this turn
        if self._dream_was_running_before_turn:
            self._dream.stop()
        
        # Always reset the idle timer on activity
        self._dream.touch()

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

    def _strip_injected_context(self, content: str) -> str:
        """Remove ephemeral memory/tool context wrappers before extraction."""
        if not isinstance(content, str):
            return ""
        content = re.sub(r"<memory-context>.*?</memory-context>", "", content, flags=re.S)
        content = re.sub(r"```memory-context.*?```", "", content, flags=re.S)
        content = re.sub(r"\[SYSTEM:[^\n]*\]", "", content)
        return content.strip()

    def _extract_session_facts(self, messages: List[Dict[str, Any]], limit: int = 5) -> list[str]:
        """Heuristic session-end fact extraction.

        This is deliberately conservative: store durable decisions, preferences,
        paths, fixes, config/cron changes, and test outcomes. Do not store raw
        chat transcripts or tool dumps.
        """
        durable_patterns = (
            r"\bremember\b", r"\bvergiss\b", r"\bnotiere\b",
            r"\bprefer\b", r"\bpreference\b", r"\balways\b", r"\bnever\b",
            r"\bwe decided\b", r"\bdecision\b", r"\buse .* instead\b",
            r"\bfixed\b", r"\bbug\b", r"\broot cause\b", r"\bworkaround\b",
            r"\bconfigured\b", r"\binstalled\b", r"\bdeployed\b", r"\bcron\b",
            r"\btests?\b", r"\bpassing\b", r"\bfailing\b",
            r"/home/", r"~/", r"\.yaml\b", r"\.py\b", r"git@", r"https?://",
        )
        facts: list[str] = []
        seen: set[str] = set()
        for m in messages:
            role = m.get("role", "")
            if role not in {"user", "assistant"}:
                continue
            text = self._strip_injected_context(m.get("content", ""))
            if not text or self._is_garbage(text):
                continue
            # Prefer bullet/short lines. Long prose gets first sentence only.
            raw_lines = []
            for line in text.splitlines():
                line = line.strip(" \t-*•>")
                if line:
                    raw_lines.append(line)
            if not raw_lines and text:
                raw_lines = [text]
            for line in raw_lines:
                if len(line) < 12 or len(line) > 900:
                    continue
                lower = line.lower()
                if any(skip in lower for skip in ("tool call", "traceback", "token usage", "<memory-context", "```json")):
                    continue
                if not any(re.search(p, line, flags=re.I) for p in durable_patterns):
                    continue
                fact = re.sub(r"\s+", " ", line).strip()
                if len(fact) > 360:
                    fact = fact[:357].rstrip() + "..."
                key = fact.lower()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(fact)
                if len(facts) >= limit:
                    return facts
        return facts

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Store compact durable session facts at session end."""
        if not self._memory or not messages:
            return
        try:
            def _cfg_bool(value, default=True):
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

            if not _cfg_bool((self._config or {}).get("session_extract_facts"), True):
                return
            limit = int((self._config or {}).get("session_fact_limit", 5) or 5)
            facts = self._extract_session_facts(messages, limit=max(1, min(limit, 8)))
            if not facts:
                return
            stored = 0
            for fact in facts:
                digest = hashlib.sha1(fact.encode("utf-8")).hexdigest()[:12]
                label = f"session-fact:{digest}"
                content = f"Session durable fact: {fact}"
                try:
                    self._memory.remember(content, label=label, auto_connect=True, detect_conflicts=True)
                    stored += 1
                except Exception as e:
                    logger.debug("Neural session fact store failed: %s", e)
            if stored:
                summary = "Session durable facts: " + " | ".join(facts[:limit])
                summary_digest = hashlib.sha1(summary.encode("utf-8")).hexdigest()[:12]
                self._memory.remember(summary, label=f"session-summary:{summary_digest}")
                logger.info("Neural memory: stored %d durable session facts", stored)
        except Exception as e:
            logger.debug("Neural on_session_end failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to neural memory. Skips garbage."""
        if action == "add" and self._memory and content:
            if self._is_garbage(content):
                return
            try:
                label = f"memory-{target}" if target else "memory"
                self._memory.remember(content, label=label)
            except Exception as e:
                logger.debug("Neural memory_write mirror failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Archive EVERYTHING before compression destroys it.

        Calls Memory.archive_compression() for full-fidelity preservation.
        This is lossless — no filtering, no "is this meaningful?" judgment.
        Every turn is stored so nothing is lost when context compresses.

        Returns a brief note for injection into context (not a replacement
        for the compressed summary — just a confirmation that archiving ran).
        """
        if not self._memory or not messages:
            return ""
        try:
            archive_raw = (self._config or {}).get("archive_raw_turns", False)
            if not (archive_raw is True or str(archive_raw).strip().lower() in {"1", "true", "yes", "on", "y"}):
                return ""
            session_tag = f"session-{self._session_id[:8]}" if self._session_id else "session-unknown"
            result = self._memory.archive_compression(
                turns=messages,
                session_tag=session_tag,
            )
            count = result.get("archived", 0)
            if count > 0:
                logger.info(f"Neural memory: archived {count} turns before compression")
            return f"[{count} conversation turns archived to neural memory before compression]"
        except Exception as e:
            logger.debug(f"Archive failed: {e}")
            return ""

    def shutdown(self) -> None:
        """Clean shutdown.

        Stops every background thread the provider owns: dream engine,
        consolidation loop, and the sponge worker. Previous code skipped
        the sponge worker entirely, so an orderly shutdown leaked the
        sponge thread on every exit (it survived only because it's a
        daemon thread, not because shutdown handled it).
        """
        # Stop dream engine
        if hasattr(self, '_dream') and self._dream:
            try:
                self._dream.stop()
            except Exception:
                pass
            self._dream = None
        # Stop consolidation
        self._consolidation_stop.set()
        if self._consolidation_thread and self._consolidation_thread.is_alive():
            self._consolidation_thread.join(timeout=2.0)
        # Stop sponge worker (was leaked previously)
        try:
            if hasattr(self, '_sponge_running') and self._sponge_running:
                self._stop_sponge()
        except Exception as e:
            logger.debug("Sponge stop during shutdown failed: %s", e)
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                pass
            self._memory = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_remember(self, args: dict) -> str:
        if self._memory is None:
            return tool_error("Neural memory provider not initialized")
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
        if self._memory is None:
            return tool_error("Neural memory provider not initialized")
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
        if self._memory is None:
            return tool_error("Neural memory provider not initialized")
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
        if self._memory is None:
            return tool_error("Neural memory provider not initialized")
        try:
            graph = self._memory.graph()
            stats = self._memory.stats()
            return json.dumps({"graph": graph, "stats": stats})
        except Exception as exc:
            return tool_error(str(exc))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config schema for `hermes memory setup neural`."""
        return [
            {
                "key": "db_path",
                "description": "Path to SQLite database file",
                "required": False,
                "default": str(Path.home() / ".neural_memory" / "memory.db"),
            },
            {
                "key": "embedding_backend",
                "description": "Embedding backend (auto, hash, tfidf, sentence-transformers)",
                "required": False,
                "default": "auto",
                "choices": ["auto", "hash", "tfidf", "sentence-transformers"],
            },
            {
                "key": "consolidation_interval",
                "description": "Background consolidation interval in seconds (0 = disabled)",
                "required": False,
                "default": 0,
            },
            {
                "key": "max_episodic",
                "description": "Maximum episodic memories (0 = unlimited)",
                "required": False,
                "default": 0,
            },
            {"key": "retrieval_mode", "description": "Retrieval mode (semantic, hybrid, advanced, skynet)", "required": False, "default": "semantic"},
            {"key": "use_hnsw", "description": "Use HNSW ANN index (auto, true, false)", "required": False, "default": "auto"},
            {"key": "lazy_graph", "description": "Hydrate graph nodes on demand instead of at startup", "required": False, "default": False},
            {"key": "think_engine", "description": "Graph thinking engine (bfs or ppr)", "required": False, "default": "bfs"},
            {"key": "rerank", "description": "Use lazy cross-encoder reranker for recall", "required": False, "default": False},
            {"key": "store_raw_turns", "description": "Store raw per-turn messages (debug only)", "required": False, "default": False},
            {"key": "archive_raw_turns", "description": "Archive raw turns before compression (debug only)", "required": False, "default": False},
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the neural memory provider with the plugin system."""
    provider = NeuralMemoryProvider()
    ctx.register_memory_provider(provider)
    
    # Activity-aware dream engine: pause while Hermes is active, resume when idle.
    # _on_pre_llm_call: stops dreaming, records state, resets idle timer
    # post_llm_call: resumes if it was running before the turn started
    ctx.register_hook("pre_llm_call", provider._on_pre_llm_call)
    ctx.register_hook("post_llm_call", provider.post_llm_call)
