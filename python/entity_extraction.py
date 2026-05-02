"""Heuristic entity extraction + EntityRegistry for unified-graph node storage.

Per Sprint 2 Phase 7 Commit 3 / handoff Section 5.1 + 5.15. Borrows entity-
intelligence patterns from Hindsight/Graphiti/Memary without splitting memory
stores: entities live as `kind='entity'` nodes in the same memories table,
linked to source memories via `mentions_entity` edges.

Heuristic-only: capitalized-word matching with stopword filter. Sufficient
for AE-domain entities (customer names, builder names, project labels). Future
commits can swap in NER/LLM extraction behind the same interface.
"""

from __future__ import annotations

import json
import re
import time as _time
from typing import Any, Optional


# Stopwords: capitalized words that are sentence-initial articles/pronouns/
# conjunctions, not entities. Conservative list — adding too many drops real
# entities. Acronyms (NEC, GFCI, EMT, MC) intentionally NOT filtered — they
# function as entities in retrieval.
_STOPWORDS: frozenset[str] = frozenset({
    # Articles + demonstratives
    "The", "This", "That", "These", "Those", "A", "An",
    # Pronouns
    "I", "We", "You", "He", "She", "It", "They", "Them",
    "My", "Your", "His", "Her", "Its", "Their", "Our",
    # Conjunctions
    "And", "Or", "But", "Yet", "So", "For", "Nor",
    # Question words
    "When", "Where", "What", "How", "Why", "Who", "Which", "Whose",
    # Conditionals + temporal connectors
    "If", "Then", "Else", "Because", "While", "After", "Before",
    "Always", "Never", "Sometimes", "Today", "Tomorrow", "Yesterday",
    # Interjections / responses
    "Yes", "No", "OK", "Okay", "Maybe", "Perhaps",
    "Now", "Soon", "Later", "Here", "There", "Then",
    # Caught post-AE-corpus-ingest 2026-05-01 — over-extracted noise:
    "NOT", "NULL", "TODO", "FIXME", "TBD", "TBA", "XXX",
    "Use", "Do", "All", "Per", "Each", "Every", "Some", "Any", "None",
    "Users",  # directory path component, not an entity
    "From", "Into", "Onto", "Upon", "Within", "Without",
    "Such", "Same", "Other", "Another", "More", "Most", "Less", "Least",
    "Just", "Only", "Even", "Also", "Still", "Yet", "Already",
    "Note", "Notes", "See", "Read", "Write", "Run", "Add", "Remove",
    # Caught in second-pass cleanup 2026-05-01 22:18Z:
    "Created", "Updated", "Deleted", "Modified", "Replaced",
    "New", "Old", "Initial", "Final", "Latest", "Previous", "Current",
    "Source", "Target", "Origin", "Destination", "Path",
    "Purpose", "Reason", "Cause", "Effect",
    "Both", "Either", "Neither",
    "Don",  # fragment from "Don't"
    "ACK", "NACK", "FYI", "RFC",  # status/protocol noise (real but generic)
    "Library",  # macOS path component
    "User", "Owner", "Author",  # generic role nouns
    "Status", "State", "Stage", "Level",
    "Type", "Kind", "Class", "Mode",
    "Default", "Auto", "Manual",
    "Block", "Step", "Item", "Entry", "Record",
    # Round-2 reviewer 2026-05-01: 38 confirmed junk entities still in top-60
    # by mention frequency. Added the high-mention ones: project meta-vocab
    # ("Phase", "Session", "Sprint"), protocol acronyms ("API", "URL", "RPC",
    # "OAuth", "HTTP", "POST", "GET", "SET", "CDP", "SQL"), workflow noise
    # ("CHECKPOINT", "MEMORY", "EOD"), single-char ("UI", "DB", "OR").
    "Phase", "Session", "Sprint", "CHECKPOINT", "MEMORY", "EOD",
    "API", "URL", "RPC", "OAuth", "HTTP", "POST", "GET", "SET",
    "CDP", "SQL", "UI", "DB", "JSON", "YAML", "TOML",
    "Commit", "Branch", "Tag", "PR", "Pull",
    "Test", "Tests", "Bench", "Audit", "Review",
})


# One or more consecutive capitalized words. Examples that match:
#   "Sarah", "Lennar", "José", "Müller", "Iñaki Choudhury", "Lot Note".
# Per Reviewer #1: extended ASCII to include Latin-1 supplement upper/lower
# ranges so Spanish-crew accented names (José/Iñaki/Müller) surface as
# entities. Cyrillic / Chinese / Greek not handled — out of scope for AE.
_UPPER_CHAR = r"[A-ZÀ-ÖØ-Þ]"
_LETTER_CHAR = r"[a-zA-ZÀ-ÖØ-öø-ÿ]"
_ENTITY_PATTERN = re.compile(
    rf"\b{_UPPER_CHAR}{_LETTER_CHAR}+(?:\s+{_UPPER_CHAR}{_LETTER_CHAR}+)*\b"
)


def extract_entities(text: str) -> list[str]:
    """Extract candidate entity names from free-text memory content.

    Returns deduplicated list preserving discovery order. Names are kept in
    their original capitalization (e.g., "Sarah", "Lennar"). Case-insensitive
    deduplication happens at registry level via _find_entity.
    """
    if not text:
        return []
    seen: set[str] = set()
    results: list[str] = []
    for match in _ENTITY_PATTERN.finditer(text):
        name = match.group(0).strip()
        if name in _STOPWORDS:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(name)
    return results


class EntityRegistry:
    """Manages entity nodes inside the unified graph (kind='entity').

    Entity nodes are rows in the `memories` table with `kind='entity'` and
    `label` set to the canonical entity name. Frequency and last_seen are
    tracked in `metadata_json`. Lookups are case-insensitive against the
    `label` column.

    Edges from source-memory → entity use `edge_type='mentions_entity'`.
    """

    def __init__(self, store: Any) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _find_entity_row(self, name: str) -> Optional[dict[str, Any]]:
        """Case-insensitive lookup by label among kind='entity' rows."""
        # Per Reviewer #1: previous version mutated conn.row_factory = None
        # globally — defensive theater (the default IS None) that would
        # silently break callers who set sqlite3.Row. Removed.
        with self.store._lock:
            row = self.store.conn.execute(
                "SELECT id, label, metadata_json FROM memories "
                "WHERE kind = 'entity' AND LOWER(label) = LOWER(?) LIMIT 1",
                (name,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "label": row[1],
            "metadata_json": row[2],
        }

    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        """Return entity dict (id, label, frequency, last_seen) or None."""
        row = self._find_entity_row(name)
        if not row:
            return None
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return {
            "id": row["id"],
            "label": row["label"],
            "frequency": meta.get("frequency", 1),
            "last_seen": meta.get("last_seen"),
        }

    def count_entities_named(self, name: str) -> int:
        with self.store._lock:
            row = self.store.conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE kind = 'entity' AND LOWER(label) = LOWER(?)",
                (name,),
            ).fetchone()
        return row[0] if row else 0

    def get_entities_for_memory(self, memory_id: int) -> list[dict[str, Any]]:
        """Return entity dicts linked from `memory_id` via mentions_entity."""
        with self.store._lock:
            rows = self.store.conn.execute(
                """SELECT m.id, m.label, m.metadata_json
                   FROM connections c
                   JOIN memories m ON m.id = c.target_id
                   WHERE c.source_id = ? AND c.edge_type = 'mentions_entity'""",
                (memory_id,),
            ).fetchall()
        out = []
        for row in rows:
            meta = json.loads(row[2]) if row[2] else {}
            out.append({
                "id": row[0],
                "label": row[1],
                "frequency": meta.get("frequency", 1),
                "last_seen": meta.get("last_seen"),
            })
        return out

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------

    def get_or_create_entity(self, name: str) -> int:
        """Return entity_id; creates if missing and increments frequency
        on existing match. Case-insensitive resolution."""
        existing = self._find_entity_row(name)
        if existing:
            self._touch_entity(existing["id"], existing["metadata_json"])
            return existing["id"]
        return self._create_entity(name)

    def _create_entity(self, name: str) -> int:
        """Create a new entity row. Frequency starts at 1."""
        # store.store() handles the typed write; we synthesize a zero-vector
        # embedding so the row is well-formed. Embedding can be backfilled by
        # later commits (Commit 6 hybrid embedding) if needed.
        zero_embedding = [0.0] * 16  # tiny placeholder; not used for entity recall
        meta = {"frequency": 1, "last_seen": _time.time()}
        return self.store.store(
            label=name,
            content=f"Entity: {name}",
            embedding=zero_embedding,
            kind="entity",
            origin_system="entity_extractor",
            metadata=meta,
        )

    def _touch_entity(self, entity_id: int, current_metadata_json: Optional[str]) -> None:
        """Increment frequency and update last_seen on an existing entity."""
        meta = json.loads(current_metadata_json) if current_metadata_json else {}
        meta["frequency"] = int(meta.get("frequency", 0)) + 1
        meta["last_seen"] = _time.time()
        with self.store._lock:
            self.store.conn.execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                (json.dumps(meta), entity_id),
            )
            self.store.conn.commit()

    def link_memory_to_entity(self, memory_id: int, entity_id: int) -> None:
        """Add a `mentions_entity` edge. Idempotent via INSERT OR REPLACE in
        SQLiteStore.add_connection."""
        self.store.add_connection(
            source=memory_id,
            target=entity_id,
            weight=1.0,
            edge_type="mentions_entity",
        )

    # ------------------------------------------------------------------
    # End-to-end retain hook
    # ------------------------------------------------------------------

    def process_memory(self, memory_id: int, text: str) -> list[int]:
        """Extract entities from text, create/link them. Returns entity IDs
        linked to the memory."""
        names = extract_entities(text)
        ids: list[int] = []
        for name in names:
            entity_id = self.get_or_create_entity(name)
            # Don't self-link: if memory_id IS the entity row (unusual but
            # possible for re-processing), skip.
            if entity_id != memory_id:
                self.link_memory_to_entity(memory_id, entity_id)
            ids.append(entity_id)
        return ids
