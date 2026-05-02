"""Constants for the unified-graph node and edge taxonomy.

Per Sprint 2 Phase 7 Commit 2 / handoff Section 13.1. These are the canonical
node kinds and edge types used by the typed retain pipeline, retrieval channels,
and dream consolidation. Keeping them as module-level frozensets makes
membership checks O(1) and prevents accidental mutation.

The vocabulary is drawn from the donor-organ model: each kind/type can be
populated by a different donor system (Hindsight world/experience/mental_model
split, Engram procedural typing, MemPalace locus, Cognee/Memify dream insights,
etc.) while the substrate stays one unified graph.
"""

from __future__ import annotations


# Memory node kinds. Default unknown when classifier cannot decide.
MEMORY_KINDS: frozenset[str] = frozenset({
    "world",            # durable external facts (NEC code, regulations, public data)
    "experience",       # events, observations, conversation turns (default)
    "mental_model",     # summaries, inferences, derived insights with evidence
    "procedural",       # how-to, workflows, conditional rules (When X, do Y)
    "entity",           # canonical entity nodes (people, customers, projects)
    "claim",            # asserted durable propositions (often supports/contradicts)
    "locus",            # memory-palace overlay nodes (rooms, wings)
    "dream_insight",    # consolidation-generated insights with derived_from edges
    "profile_trait",    # user/system profile traits with evidence provenance
    "memory_summary",   # compressed summary nodes preserving evidence links
    "benchmark_trace",  # benchmark run traces for retrievable failure analysis
    "unknown",          # fallback when classifier cannot decide
})


# Edge types between nodes in the unified graph.
# Round-3 reviewer 2026-05-01: registry was out of sync with code/DB.
# - "similar" had 1.2M live rows but was missing from registry
# - "applies_to" referenced by intent-weight maps + tests but missing
# - "bridges" was canonical name; live DB used "rem_bridge" (1 row)
# - 6 entries are aspirational (zero writers); kept as forward placeholders
# is_valid_edge_type() is logging-only — does not gate writes (would
# break the 1.2M `similar` default path mid-flight).
EDGE_TYPES: frozenset[str] = frozenset({
    # Active in live DB (write paths exist):
    "similar",               # auto-similar default (1.2M+ rows, the dominant edge)
    "mentions_entity",       # memory → entity reference (~23K rows)
    "rem_bridge",            # REM-bridging edge between weakly-linked nodes (legacy "bridges")
    "summarizes",            # summary node → source memories (D5/Phase 7 Commit 9)
    "derived_from",          # insight → supporting evidence (Phase 7 Commit 4)
    "located_in",            # memory → locus / locus → wing (Phase 7 Commit 10)
    "contradicts",           # one claim contradicts another (Phase 7 Commit 9)
    "applies_to",            # used in intent-weight maps + tests
    # Phase 7 spec — currently no writers but keep as forward placeholders:
    "semantic_similar_to",   # cosine/embedding similarity (alias path)
    "happened_before",       # temporal ordering
    "caused_by",             # causal chain
    "supports",              # one claim supports another
    "promotes_to",           # working → durable promotion
    "decays_from",           # decay-chain provenance
    "validated_by",          # link confirmed by retrieval/dream reinforcement
    "invalidated_by",        # link superseded or contradicted
    "retrieved_for",         # link to historic queries that surfaced this memory
    "failed_on",             # benchmark failure linkage
})


def is_valid_kind(kind: str) -> bool:
    """Return True if `kind` is a recognized memory kind."""
    return kind in MEMORY_KINDS


def is_valid_edge_type(edge_type: str) -> bool:
    """Return True if `edge_type` is a recognized edge type."""
    return edge_type in EDGE_TYPES
