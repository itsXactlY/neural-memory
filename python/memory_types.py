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
EDGE_TYPES: frozenset[str] = frozenset({
    "semantic_similar_to",   # cosine/embedding similarity
    "mentions_entity",       # memory → entity reference
    "happened_before",       # temporal ordering
    "caused_by",             # causal chain
    "supports",              # one claim supports another
    "contradicts",           # one claim contradicts another
    "summarizes",            # summary node → source memories
    "derived_from",          # insight → supporting evidence
    "located_in",            # memory → locus / locus → wing
    "bridges",               # REM-bridging edge between weakly-linked nodes
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
