"""
Mazemaker Benchmark — Dataset Generator
============================================
Generates synthetic but realistic memory datasets for benchmarking.
All data is self-contained — no external downloads required.

Dataset types:
  - episodic:    Agent session memories (what happened, when, outcome)
  - factual:     Knowledge base facts (entities, relationships)
  - temporal:    Time-series events with timestamps
  - conversational: Chat logs between user and agent
  - graph:       Interconnected knowledge nodes
  - adversarial:  Edge cases: near-duplicates, conflicts, poison data
"""
import hashlib
import json
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ── Utility ──────────────────────────────────────────────────────────────────

def seeded(seed: int):
    """Return a seeded random.Random instance."""
    r = random.Random()
    r.seed(seed)
    return r


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def sliding_window(items: List, window: int = 3):
    """Yield sliding windows of size `window`."""
    for i in range(len(items) - window + 1):
        yield items[i:i + window]


# ── Base Generator ────────────────────────────────────────────────────────────

class BaseGenerator(ABC):
    """Abstract base for all memory generators."""

    def __init__(self, seed: int = 42):
        self.rng = seeded(seed)

    @abstractmethod
    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        """Yield `count` memory records.

        Each record must have at minimum:
          - id:       unique identifier (string)
          - text:    the memory text (string)
          - label:   a semantic label (string)
          - metadata: extra structured data (dict)
        """
        ...

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        return list(self.generate(count))


# ── Epidosic Generator ───────────────────────────────────────────────────────

class EpisodicGenerator(BaseGenerator):
    """
    Simulates agent session memories — what the agent did, why, and the outcome.
    Covers: file edits, tool calls, decisions, errors, recoveries.

    Memory structure:
      - action: what was done
      - context: why it was needed
      - outcome: success/failure
      - entities: files, tools, concepts involved
    """

    ACTIONS = [
        ("edited", "file {path}", "Applied {change} to {path}"),
        ("created", "file {path}", "Created new file {path}"),
        ("deleted", "file {path}", "Removed file {path}"),
        ("ran", "command {cmd}", "Executed command: {cmd}"),
        ("fixed", "bug in {component}", "Resolved bug in {component}"),
        ("deployed", "service {name}", "Deployed {name} to production"),
        ("refactored", "module {module}", "Refactored {module} for better structure"),
        ("tested", "feature {feature}", "Ran tests for {feature}"),
        ("reviewed", "code {file}", "Code review: {file}"),
        ("configured", "system {system}", "Configured {system} settings"),
        ("debugged", "issue {issue}", "Debugged and resolved {issue}"),
        ("optimized", "performance of {target}", "Improved performance of {target}"),
        ("documented", "API {api}", "Wrote documentation for {api}"),
        ("migrated", "database schema", "Migrated database schema for {reason}"),
        ("integrated", "service {service}", "Integrated {service} via API"),
    ]

    COMPONENTS = [
        "memory_client.py", "embed_provider.py", "dream_engine.py", "gpu_recall.py",
        "neural_memory.py", "sync_bridge.py", "test_suite.py", "dashboard.py",
        "auth.py", "config.py", "api.py", "router.py", "cache.py", "indexer.py",
    ]

    PATHS = [
        "~/projects/mazemaker/python/", "~/projects/pulse/api/",
        "~/.hermes/plugins/memory/neural/", "/tmp/build/output/",
        "~/projects/remainder-website/src/", "~/projects/haus-suche/scraper/",
    ]

    COMMANDS = [
        "python3 test_suite.py --tags embed,memory",
        "bash install.sh install",
        "git push origin main",
        "curl -X POST /api/ingest",
        "pytest -v tests/",
        "cmake --build . -j$(nproc)",
        "rsync -avz src/ dest/",
        "docker build -t app .",
        "kubectl apply -f deployment.yaml",
    ]

    OUTCOMES = [
        ("succeeded", "The operation completed successfully without errors."),
        ("failed", "The operation failed due to a permission error."),
        ("succeeded_with_warnings", "Completed but generated warnings in output."),
        ("partially_completed", "Some steps succeeded, others were skipped."),
        ("timed_out", "Operation exceeded the configured timeout."),
        ("rolled_back", "Changes were rolled back after a critical error."),
    ]

    def __init__(self, seed: int = 42, count: int = 5000):
        super().__init__(seed)
        self.count = count
        self.templates = self._build_templates()

    def _build_templates(self) -> List[Dict]:
        templates = []
        for verb, entity, description in self.ACTIONS:
            if "{path}" in description:
                template = description.replace("{path}", "{entity}")
            elif "{cmd}" in description:
                template = description.replace("{cmd}", "{entity}")
            elif "{api}" in description:
                template = description.replace("{api}", "{entity}")
            else:
                template = description.replace("{target}", "{entity}").replace(
                    "{issue}", "{entity}"
                )
            templates.append({
                "verb": verb,
                "entity": entity,
                "template": template,
            })
        return templates

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        for i in range(count):
            tmpl = self.rng.choice(self.templates)
            entity = self._pick_entity(tmpl["verb"])
            action_id = f"episodic-{i:06d}-{sha256(tmpl['template']+entity)[:6]}"

            outcome_verb, outcome_desc = self.rng.choices(
                self.OUTCOMES, weights=[7, 1, 1, 0.5, 0.3, 0.2]
            )[0]

            text = (
                f"Agent {tmpl['verb']} {entity}. "
                f"Reason: {self._random_reason()}. "
                f"Outcome: {outcome_desc}"
            )

            yield {
                "id": action_id,
                "text": text,
                "label": f"episodic:{tmpl['verb']}",
                "metadata": {
                    "type": "episodic",
                    "verb": tmpl["verb"],
                    "entity": entity,
                    "outcome": outcome_verb,
                    "reason": self._random_reason(),
                    "session_id": self.rng.randint(1, 500),
                    "turn": self.rng.randint(1, 200),
                },
            }

    def _pick_entity(self, verb: str) -> str:
        if verb in ("edited", "created", "deleted"):
            return self.rng.choice(self.COMPONENTS)
        elif verb == "ran":
            return self.rng.choice(self.COMMANDS)
        elif verb in ("configured", "integrated"):
            return self.rng.choice([
                "memory provider", "embedding backend", "Discord webhook",
                "Postgres bridge", "FastAPI server", "nginx reverse proxy",
            ])
        else:
            return self.rng.choice(self.COMPONENTS)

    def _random_reason(self) -> str:
        reasons = [
            "User requested this change explicitly.",
            "Bug report filed by user indicated this was broken.",
            "Performance profiling showed this was a bottleneck.",
            "Code review feedback required this modification.",
            "Dependency update necessitated this adjustment.",
            "Security audit flagged this as a risk.",
            "Automated test failure triggered this investigation.",
            "User preference changed and needed updating.",
            "New feature requirement from the roadmap.",
            "Refactoring needed to support future changes.",
        ]
        return self.rng.choice(reasons)


# ── Factual Generator ─────────────────────────────────────────────────────────

class FactualGenerator(BaseGenerator):
    """
    Knowledge base facts: entities, relationships, attributes.
    Tests semantic recall and cross-entity connections.

    Structure:
      - entity: the subject
      - predicate: the relationship/attribute
      - object: the value
    """

    def __init__(self, seed: int = 42, count: int = 3000):
        super().__init__(seed)
        self.count = count

    ENTITIES = [
        ("Mazemaker", "is a", "persistent memory system for AI agents"),
        ("Mazemaker", "supports", "semantic, hybrid, and advanced retrieval modes"),
        ("Mazemaker", "uses", "BAAI/bge-m3 embeddings (1024d)"),
        ("Mazemaker", "has a", "Dream Engine for autonomous consolidation"),
        ("Mazemaker", "stores memories in", "SQLite with WAL mode"),
        ("FastEmbed", "is", "an ONNX-based embedding library"),
        ("FastEmbed", "loads in", "under 1 second on modern hardware"),
        ("DreamEngine", "has", "three phases: NREM, REM, Insight"),
        ("DreamEngine", "runs during", "idle periods between LLM calls"),
        ("Hermes", "is built by", "Nous Research"),
        ("Hermes", "has", "Mazemaker as a plugin"),
        ("HNSW", "is a", "graph-based ANN index"),
        ("HNSW", "activates automatically when", "memory count exceeds threshold"),
        ("PPR", "stands for", "Personalized PageRank"),
        ("PPR", "is used for", "spreading activation recall"),
        ("RRF", "stands for", "Reciprocal Rank Fusion"),
        ("RRF", "combines", "multiple retrieval signals"),
        ("WAL", "stands for", "Write-Ahead Log"),
        ("WAL", "enables", "concurrent reads during writes"),
        ("Postgres", "is used as", "cold storage for neural memories"),
        ("BGE-M3", "is a", "state-of-the-art multilingual embedding model"),
        ("Louvain", "is a", "community detection algorithm"),
        ("Salience", "decays over", "time based on access frequency"),
        ("Conflict detection", "marks", "contradictory memories as [SUPERSEDED]"),
        ("Supersession", "records", "the revision chain in memory_revisions"),
    ]

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        # Cycle through entity groups with variations
        for i in range(count):
            base_idx = i % len(self.ENTITIES)
            entity, predicate, object_ = self.ENTITIES[base_idx]

            # Add variations to make each unique
            variation = i // len(self.ENTITIES)
            if variation > 0:
                qualifiers = [
                    f"at scale tier {variation}",
                    f"in project {chr(65+variation)}",
                    f"after {variation} iterations",
                    f"with {variation} concurrent users",
                ]
                extra = self.rng.choice(qualifiers) if self.rng.random() > 0.5 else ""
                text = f"{entity} {predicate} {object_}. {extra}".strip(". ")
            else:
                text = f"{entity} {predicate} {object_}."

            # Generate a related but different fact for diversity
            if self.rng.random() > 0.7:
                alt_entity = self.rng.choice([e for e, _, _ in self.ENTITIES])
                text += f" Related: {alt_entity} is also relevant to this context."

            yield {
                "id": f"factual-{i:06d}-{sha256(text)[:6]}",
                "text": text,
                "label": f"factual:{entity.lower().replace(' ', '_')}",
                "metadata": {
                    "type": "factual",
                    "entity": entity,
                    "predicate": predicate,
                    "object": object_,
                    "variation": i // len(self.ENTITIES),
                },
            }


# ── Temporal Generator ────────────────────────────────────────────────────────

class TemporalGenerator(BaseGenerator):
    """
    Time-series events with timestamps.
    Tests temporal decay, recency weighting, and time-range queries.
    """

    EVENT_TYPES = [
        ("user_login", "User {user} logged in from {location}"),
        ("file_modified", "File {file} was modified (size: {size} bytes)"),
        ("error_occurred", "Error: {error_type} in {component}"),
        ("deployment", "Deployment {name} started/completed/failed"),
        ("memory_recalled", "Memory recall triggered: query={query_type}"),
        ("dream_cycle", "Dream engine cycle {phase} completed"),
        ("model_switched", "Model switched to {model} via {provider}"),
        ("api_call", "API call to {endpoint} returned {status}"),
        ("config_changed", "Config key {key} changed from {old} to {new}"),
        ("memory_stored", "New memory stored: {category}, salience={salience:.2f}"),
    ]

    LOCATIONS = ["Berlin", "Munich", "Frankfurt", "Hamburg", "Cologne", "Remote/VPN"]
    USERS = ["alca", "hermes", "dev", "admin", "cron-agent"]
    ERROR_TYPES = ["TimeoutError", "PermissionError", "ConnectionRefused", "KeyError", "ValueError"]
    COMPONENTS = ["neural_memory.py", "embed_provider.py", "api.py", "auth.py", "router.py"]
    MODELS = ["gpt-4o", "claude-sonnet-4", "gemini-2.0-flash", "llama-3.3-70b"]
    PROVIDERS = ["openai", "anthropic", "google", "openrouter"]
    ENDPOINTS = ["/api/ingest", "/api/recall", "/api/search", "/api/sync"]
    STATUS_CODES = ["200", "201", "400", "404", "429", "500"]

    def __init__(self, seed: int = 42, count: int = 2000):
        super().__init__(seed)
        self.count = count
        self.base_time = datetime(2026, 1, 1, 0, 0, 0)

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        for i in range(count):
            event_type, template = self.rng.choice(self.EVENT_TYPES)
            text = self._render_template(template)

            # Spread events over ~6 months
            delta = timedelta(
                seconds=self.rng.randint(0, 6 * 30 * 24 * 3600)
            )
            ts = self.base_time + delta

            yield {
                "id": f"temporal-{i:06d}-{sha256(text)[:6]}",
                "text": text,
                "label": f"temporal:{event_type}",
                "metadata": {
                    "type": "temporal",
                    "event_type": event_type,
                    "timestamp": ts.isoformat(),
                    "unix_ts": ts.timestamp(),
                },
            }

    def _render_template(self, template: str) -> str:
        replacements = {
            "{user}": self.rng.choice(self.USERS),
            "{location}": self.rng.choice(self.LOCATIONS),
            "{file}": self.rng.choice(self.COMPONENTS),
            "{error_type}": self.rng.choice(self.ERROR_TYPES),
            "{component}": self.rng.choice(self.COMPONENTS),
            "{name}": f"deploy-{self.rng.randint(1,100)}",
            "{query_type}": self.rng.choice(["semantic", "hybrid", "temporal", "graph"]),
            "{phase}": self.rng.choice(["nrem", "rem", "insight"]),
            "{model}": self.rng.choice(self.MODELS),
            "{provider}": self.rng.choice(self.PROVIDERS),
            "{endpoint}": self.rng.choice(self.ENDPOINTS),
            "{status}": self.rng.choice(self.STATUS_CODES),
            "{key}": self.rng.choice(["retrieval_mode", "embedding_backend", "max_connections"]),
            "{old}": str(self.rng.randint(0, 100)),
            "{new}": str(self.rng.randint(0, 100)),
            "{size}": str(self.rng.randint(100, 1_000_000)),
            "{salience}": str(round(self.rng.uniform(0.1, 2.5), 2)),
        }
        text = template
        for k, v in replacements.items():
            if k in text:
                text = text.replace(k, v)
        return text


# ── Conversational Generator ──────────────────────────────────────────────────

class ConversationalGenerator(BaseGenerator):
    """
    Simulates chat logs between user and agent.
    Tests recall of specific conversations, topics, and decisions.
    """

    def __init__(self, seed: int = 42, count: int = 1000):
        super().__init__(seed)
        self.count = count

    USER_INTENTS = [
        "debug my mazemaker",
        "run the benchmark suite",
        "check the dream engine status",
        "deploy the API to production",
        "fix the HNSW index corruption",
        "add a new memory source",
        "configure GPU recall",
        "review the latest commits",
        "update the embedding model",
        "investigate slow query latency",
        "create a new cron job",
        "verify Postgres sync",
        "optimize retrieval speed",
        "write tests for the benchmark",
        "analyze memory fragmentation",
    ]

    AGENT_ACTIONS = [
        "Investigated the issue, found root cause in embed_provider.py line 142.",
        "Ran diagnostics: GPU memory at 87%, CPU at 23%. Recommendation: increase GPU batch size.",
        "Applied fix: changed retrieval_mode from 'semantic' to 'hybrid'.",
        "Benchmark results: retrieval latency improved by 34% after HNSW activation.",
        "Dream cycle completed: 142 connections strengthened, 23 pruned.",
        "Postgres sync completed: 5,000 records transferred in 12.3s.",
        "Deployed v2.4.1 to production. Health checks passing.",
        "Refactored memory_client.py: extracted embedding logic into embed_provider.",
        "Tests written and passing: 47/47 unit tests, 12/12 integration tests.",
        "Config updated: max_connections=50, hnsw_threshold=5000, salience_decay=0.95.",
    ]

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        session_id = 1
        turn = 1
        current_session_turns = 0

        for i in range(count):
            if current_session_turns >= self.rng.randint(5, 25):
                session_id += 1
                turn = 1
                current_session_turns = 0

            is_user = self.rng.random() > 0.4
            if is_user:
                text = self.rng.choice(self.USER_INTENTS)
                speaker = "user"
            else:
                text = self.rng.choice(self.AGENT_ACTIONS)
                speaker = "agent"

            current_session_turns += 1

            yield {
                "id": f"conv-{session_id:04d}-{turn:03d}-{sha256(text)[:6]}",
                "text": f"[Session {session_id}] {speaker}: {text}",
                "label": f"conversational:session_{session_id}",
                "metadata": {
                    "type": "conversational",
                    "session_id": session_id,
                    "turn": turn,
                    "speaker": speaker,
                    "intent": text if is_user else None,
                },
            }
            turn += 1


# ── Graph Generator ───────────────────────────────────────────────────────────

class GraphGenerator(BaseGenerator):
    """
    Generates interconnected knowledge nodes with known relationships.
    Tests graph traversal (BFS, PPR), community detection, and bridging recall.
    """

    def __init__(self, seed: int = 42, count: int = 500):
        super().__init__(seed)
        self.count = count

    KNOWLEDGE_CLUSTERS = {
        "memory": [
            "Mazemaker provides persistent memory for AI agents",
            "Memory embeddings use BAAI/bge-m3 (1024 dimensions)",
            "FastEmbed loads embeddings in under 1 second",
            "Salience tracking assigns importance scores to memories",
            "The Dream Engine consolidates memories during idle periods",
            "HNSW index activates automatically above 5000 memories",
            "Conflict detection marks contradictory memories as [SUPERSEDED]",
            "The graph layer stores connections between related memories",
            "Spreading activation traverses the memory graph",
            "PPR enables personalized recall based on access patterns",
        ],
        "infrastructure": [
            "Hermes Agent runs on Python 3.10+ with asyncio",
            "The Mazemaker plugin lives in ~/.hermes/plugins/memory/neural/",
            "Cron jobs run autonomous tasks every hour",
            "Discord webhooks notify on system events",
            "The API server runs on port 8443 with TLS",
            "SQLite stores memories in WAL mode for concurrent access",
            "Postgres serves as cold storage for archived memories",
            "The sync bridge transfers data from SQLite to Postgres",
            "WAL checkpoints run automatically every 5 seconds",
            "Backup repo syncs to github.com/itsXactlY/hermes-backup",
        ],
        "models": [
            "BAAI/bge-m3 is the primary embedding model (multilingual, 1024d)",
            "sentence-transformers is the fallback embedding backend",
            "GPU recall uses torch.matmul for batch cosine similarity",
            "The shared embedding server uses UNIX domain sockets",
            "Model loading waits up to 120 seconds for GPU availability",
            "CPU fallback is disabled — GPU is mandatory",
            "The embed server auto-ejects GPU after idle timeout",
            "FastEmbed ONNX model lives in ~/.neural_memory/models/",
        ],
        "benchmarks": [
            "MemoryAgentBench tests recall accuracy across categories",
            "EvoMem is a streaming benchmark for self-evolving memory",
            "LongMemEval tests long-range understanding (32k+ context)",
            "The Mazemaker benchmark generates synthetic datasets",
            "Retrieval quality is measured via Recall@k and MRR",
            "Throughput is measured as embeddings/second",
            "Latency benchmarks measure p50, p95, p99 percentiles",
            "Concurrent benchmarks stress-test WAL under multi-threaded load",
        ],
        "pulse": [
            "PULSE is a multi-source social search engine",
            "PULSE aggregates Reddit, HN, Lobste.rs, Product Hunt",
            "PULSE uses engagement scoring (karma, comments, forks)",
            "PULSE findings need content-hash dedup before neural ingestion",
            "The PULSE API runs at remainder.online",
            "Demo key 'demo' only allows curated prompts",
        ],
    }

    # Cross-cluster connections (bridges between topics)
    CROSS_CLUSTERS = [
        ("memory", "benchmarks", "MemoryAgentBench tests Mazemaker recall quality"),
        ("memory", "infrastructure", "Mazemaker stores benchmark results as memories"),
        ("benchmarks", "models", "GPU recall is benchmarked against CPU baseline"),
        ("pulse", "infrastructure", "PULSE findings are stored via the sync bridge"),
        ("memory", "models", "BGE-M3 embeddings power Mazemaker retrieval"),
        ("pulse", "benchmarks", "PULSE search quality is measured by recall@5"),
    ]

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        used = 0
        for cluster_name, facts in self.KNOWLEDGE_CLUSTERS.items():
            for i, fact in enumerate(facts):
                if used >= count:
                    return
                yield {
                    "id": f"graph-{cluster_name}-{i:02d}-{sha256(fact)[:6]}",
                    "text": fact,
                    "label": f"graph:{cluster_name}",
                    "metadata": {
                        "type": "graph",
                        "cluster": cluster_name,
                        "index_in_cluster": i,
                        "is_bridge": False,
                    },
                }
                used += 1

        # Add cross-cluster bridges
        for source, target, fact in self.CROSS_CLUSTERS:
            if used >= count:
                return
            yield {
                "id": f"graph-bridge-{source}-{target}-{sha256(fact)[:6]}",
                "text": fact,
                "label": f"graph:bridge_{source}_{target}",
                "metadata": {
                    "type": "graph",
                    "cluster": "bridge",
                    "source_cluster": source,
                    "target_cluster": target,
                    "is_bridge": True,
                },
            }
            used += 1

        # Fill remaining with cluster-extended facts
        while used < count:
            cluster = self.rng.choice(list(self.KNOWLEDGE_CLUSTERS.keys()))
            base = self.rng.choice(self.KNOWLEDGE_CLUSTERS[cluster])
            text = f"{base} This knowledge is part of the {cluster} cluster."
            yield {
                "id": f"graph-ext-{used:06d}-{sha256(text)[:6]}",
                "text": text,
                "label": f"graph:{cluster}",
                "metadata": {
                    "type": "graph",
                    "cluster": cluster,
                    "is_bridge": False,
                },
            }
            used += 1


# ── Adversarial Generator ─────────────────────────────────────────────────────

class AdversarialGenerator(BaseGenerator):
    """
    Edge cases: near-duplicates, direct conflicts, poison data, empty edge cases.
    Tests the robustness of conflict detection, salience decay, and retrieval precision.
    """

    def __init__(self, seed: int = 42, count: int = 500):
        super().__init__(seed)
        self.count = count

    CONFLICT_PAIRS = [
        ("FastEmbed loads embeddings in under 1 second.",  # original
         "FastEmbed loads embeddings in 4-5 seconds on average."),  # conflict
        ("The Dream Engine runs NREM consolidation first.",
         "The Dream Engine runs REM consolidation before NREM."),
        ("HNSW activates above 5,000 memories.",
         "HNSW activates above 10,000 memories."),
        ("Salience decays with a factor of 0.95 per day.",
         "Salience decays with a factor of 0.5 per day."),
        ("GPU recall uses torch.matmul for batch similarity.",
         "GPU recall uses cosine similarity via sklearn."),
        ("WAL mode allows concurrent reads during writes.",
         "WAL mode blocks all reads during write transactions."),
        ("Postgres bridge syncs one-way from SQLite to Postgres.",
         "Postgres bridge syncs bidirectionally between SQLite and Postgres."),
        ("The embed server uses UNIX domain sockets.",
         "The embed server uses TCP localhost:8080."),
    ]

    NEAR_DUPLICATE_TEMPLATES = [
        ("The memory store uses SQLite with WAL mode.", [
            "The memory store uses SQLite WAL mode for performance.",
            "SQLite with WAL mode serves as the memory store.",
            "WAL mode SQLite is used as the memory store backend.",
            "The store: SQLite in WAL mode for memory persistence.",
            "Memory persistence via SQLite WAL mode.",
        ]),
        ("BGE-M3 produces 1024-dimensional embeddings.", [
            "BGE-M3 outputs 1024-dimensional vectors.",
            "The BGE-M3 model generates 1024d embeddings.",
            "Embeddings from BGE-M3 have 1024 dimensions.",
            "1024d embeddings come from BGE-M3.",
        ]),
    ]

    def generate(self, count: int) -> Generator[Dict[str, Any], None, None]:
        generated = 0

        # 1. Conflict pairs (50%)
        num_conflicts = count // 2
        for i in range(num_conflicts):
            orig, conflict = self.rng.choice(self.CONFLICT_PAIRS)
            # Alternate between original and conflicting
            if i % 2 == 0:
                text = orig
                label = "adversarial:original"
                is_conflict = False
            else:
                text = conflict
                label = "adversarial:conflict"
                is_conflict = True

            yield {
                "id": f"adv-conflict-{i:05d}-{sha256(text)[:6]}",
                "text": text,
                "label": label,
                "metadata": {
                    "type": "adversarial",
                    "subtype": "conflict",
                    "is_conflict": is_conflict,
                    "conflict_group": i // 2,
                },
            }
            generated += 1

        # 2. Near-duplicates (30%)
        num_near_dup = int(count * 0.3)
        for i in range(num_near_dup):
            template, variants = self.rng.choice(self.NEAR_DUPLICATE_TEMPLATES)
            text = self.rng.choice(variants)
            yield {
                "id": f"adv-near-dup-{i:05d}-{sha256(text)[:6]}",
                "text": text,
                "label": "adversarial:near_duplicate",
                "metadata": {
                    "type": "adversarial",
                    "subtype": "near_duplicate",
                    "template": template[:50],
                },
            }
            generated += 1

        # 3. Edge cases: very short, very long, special chars, unicode
        num_edge = count - generated
        edge_cases = [
            ("", "adversarial:empty"),  # empty string
            ("   ", "adversarial:whitespace"),  # whitespace only
            ("🤖🧠💾", "adversarial:emoji"),  # emoji only
            ("SELECT * FROM memories; DROP TABLE memories;", "adversarial:injection"),  # injection attempt
            ("A", "adversarial:minimal"),  # minimal content
            ("x" * 5000, "adversarial:very_long"),  # very long
            ("\n\t\r", "adversarial:control_chars"),  # control chars
            ("Hérmës Agént ✓", "adversarial:unicode"),  # unicode
        ]
        for i, (text, label) in enumerate(edge_cases[:num_edge]):
            yield {
                "id": f"adv-edge-{i:05d}-{sha256(text or 'empty')[:6]}",
                "text": text,
                "label": label,
                "metadata": {
                    "type": "adversarial",
                    "subtype": label.split(":")[1],
                },
            }


# ── Query Generator ────────────────────────────────────────────────────────────

class QueryGenerator:
    """
    Generates ground-truth queries from stored memories.
    Each query has a known set of relevant memories for recall measurement.
    """

    def __init__(self, memories: List[Dict[str, Any]], seed: int = 42):
        self.rng = seeded(seed)
        self.memories = memories
        self.by_label = self._index_by_label()

    def _index_by_label(self) -> Dict[str, List[Dict]]:
        by_label = {}
        for m in self.memories:
            label = m.get("label", "unknown")
            by_label.setdefault(label, []).append(m)
        return by_label

    def generate_recall_queries(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate queries with known ground-truth answers."""
        queries = []
        for _ in range(count):
            if not self.by_label:
                break
            label = self.rng.choice(list(self.by_label.keys()))
            candidates = self.by_label[label]
            if not candidates:
                continue
            # Pick a memory to form a query from
            source = self.rng.choice(candidates)
            text = source["text"]
            # Create query by extracting key phrases
            words = text.split()
            if len(words) > 5:
                query_words = self.rng.sample(words, min(5, len(words)))
                query_text = " ".join(query_words)
            else:
                query_text = text[:60]

            queries.append({
                "query": query_text,
                "label": label,
                "ground_truth_ids": [m["id"] for m in candidates],
                "num_relevant": len(candidates),
            })
        return queries

    def generate_temporal_queries(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate time-range and recency queries."""
        temporal_memories = [
            m for m in self.memories
            if m.get("metadata", {}).get("type") == "temporal"
        ]
        if not temporal_memories:
            return []
        queries = []
        for _ in range(count):
            m = self.rng.choice(temporal_memories)
            ts = m.get("metadata", {}).get("unix_ts", time.time())
            queries.append({
                "query": m["text"][:60],
                "timestamp": ts,
                "ground_truth_id": m["id"],
            })
        return queries


# ── Master Dataset ────────────────────────────────────────────────────────────

class MasterDataset:
    """
    Generates the complete benchmark dataset from all generators.
    Call `.generate()` once to get all memories, then use QueryGenerator
    for recall queries.
    """

    GENERATORS = {
        "episodic": EpisodicGenerator,
        "factual": FactualGenerator,
        "temporal": TemporalGenerator,
        "conversational": ConversationalGenerator,
        "graph": GraphGenerator,
        "adversarial": AdversarialGenerator,
    }

    def __init__(self, config=None, seed: int = 42):
        self.seed = seed
        self.config = config or {}
        self.rng = seeded(seed)

    def generate(
        self,
        episodic: int = 5000,
        factual: int = 3000,
        temporal: int = 2000,
        conversational: int = 1000,
        graph: int = 500,
        adversarial: int = 500,
    ) -> List[Dict[str, Any]]:
        """Generate all memories and return as a flat list."""
        all_memories = []
        generators = {
            "episodic": EpisodicGenerator(seed=self.seed, count=episodic),
            "factual": FactualGenerator(seed=self.seed + 1),
            "temporal": TemporalGenerator(seed=self.seed + 2, count=temporal),
            "conversational": ConversationalGenerator(seed=self.seed + 3),
            "graph": GraphGenerator(seed=self.seed + 4),
            "adversarial": AdversarialGenerator(seed=self.seed + 5),
        }
        for name, gen in generators.items():
            count = locals()[name]
            for memory in gen.generate(count):
                all_memories.append(memory)

        self.rng.shuffle(all_memories)
        return all_memories

    def generate_scales(self) -> Dict[int, List[Dict[str, Any]]]:
        """Generate memories at all configured scale tiers."""
        # First generate max-scale dataset
        total = sum([
            self.config.get("episodic", 5000),
            self.config.get("factual", 3000),
            self.config.get("temporal", 2000),
            self.config.get("conversational", 1000),
            self.config.get("graph", 500),
            self.config.get("adversarial", 500),
        ])
        all_memories = self.generate(
            episodic=5000, factual=3000, temporal=2000,
            conversational=1000, graph=500, adversarial=500,
        )

        scales = {}
        tiers = self.config.get("scale_tiers", [100, 1_000, 10_000, 50_000, 100_000])
        for tier in sorted(set(tiers)):
            if tier <= len(all_memories):
                scales[tier] = all_memories[:tier]
            else:
                # Scale up by cycling through generated data
                repeats = (tier // len(all_memories)) + 1
                scaled = (all_memories * repeats)[:tier]
                # Add unique IDs to avoid collisions
                for i, mem in enumerate(scaled):
                    mem = dict(mem)
                    mem["id"] = f"scaled-{tier}-{i:07d}-{sha256(mem['text'])[:6]}"
                    scaled[i] = mem
                scales[tier] = scaled
        return scales


# ── CLI helpers ───────────────────────────────────────────────────────────────

def load_or_generate_dataset(config=None, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Load from cache or generate fresh dataset.
    Returns (memories, queries).
    """
    cfg = config or {}
    cache_path = Path(cfg.get("cache_path", "~/.neural_memory_benchmark/data/dataset.json")).expanduser()

    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        memories = data["memories"]
        queries = data["queries"]
        return memories, queries

    # Generate fresh
    ds = MasterDataset(seed=seed)
    memories = ds.generate(
        episodic=cfg.get("episodic", 5000),
        factual=cfg.get("factual", 3000),
        temporal=cfg.get("temporal", 2000),
        conversational=cfg.get("conversational", 1000),
        graph=cfg.get("graph", 500),
        adversarial=cfg.get("adversarial", 500),
    )

    qgen = QueryGenerator(memories, seed=seed)
    queries = qgen.generate_recall_queries(count=cfg.get("queries", 50))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"memories": memories, "queries": queries}, indent=2))
    return memories, queries


if __name__ == "__main__":
    ds = MasterDataset(seed=42)
    memories = ds.generate()
    print(f"Generated {len(memories)} memories:")
    from collections import Counter
    counts = Counter(m["label"].split(":")[0] for m in memories)
    for k, v in counts.items():
        print(f"  {k}: {v}")
