"""Heuristic memory-kind classifier.

Maps free-text memory content to one of the kinds in `memory_types.MEMORY_KINDS`.
Heuristic-only — no LLM, no model load, no network. The function is fast (~10us
per call) and deterministic, suitable for the retain hot path.

Future commits may add an LLM-backed classifier behind the same function
signature; the heuristic remains as a fallback when an LLM is unavailable.

Decision order (first match wins):
    1. procedural — imperative how-to / workflow / conditional rules
    2. world      — durable external/regulatory facts
    3. mental_model — summary, inference, conclusion language
    4. experience — default; event/observation language

Per Sprint 2 Phase 7 Commit 2 / handoff Section 8.2.
"""

from __future__ import annotations

import re
from typing import Any

from memory_types import MEMORY_KINDS


# Procedural patterns: imperative how-to / conditional rules / workflow steps.
# AE-domain examples that should classify procedural:
#   "When estimating panel upgrades, check load calc first."
#   "If permit jurisdiction is unclear, call the village before quoting."
#   "Always photograph panel labels before leaving the job."
_PROCEDURAL_PATTERNS = [
    # English imperative / conditional / how-to patterns
    re.compile(r"^\s*when\s+\w", re.IGNORECASE),
    re.compile(r"^\s*if\s+\w", re.IGNORECASE),
    re.compile(r"^\s*always\b", re.IGNORECASE),
    re.compile(r"^\s*never\b", re.IGNORECASE),
    re.compile(r"^\s*before\s+\w", re.IGNORECASE),
    re.compile(r"^\s*after\s+\w", re.IGNORECASE),
    re.compile(r"\bhow\s+(do|to|should)\s+(we|you|i)\b", re.IGNORECASE),
    re.compile(r"\bsteps?\s+(to|for)\b", re.IGNORECASE),
    re.compile(r"\bcheck(?:list)?\b.*\bbefore\b", re.IGNORECASE),
    re.compile(r"\b(remember|make sure)\s+to\b", re.IGNORECASE),
    # Spanish patterns for AE crew comms (whatsapp Spanish messages)
    re.compile(r"^\s*cuando\s+\w", re.IGNORECASE),         # "When..."
    re.compile(r"^\s*si\s+\w", re.IGNORECASE),             # "If..."
    re.compile(r"^\s*siempre\b", re.IGNORECASE),           # "Always..."
    re.compile(r"^\s*nunca\b", re.IGNORECASE),             # "Never..."
    re.compile(r"^\s*antes\s+de\s+\w", re.IGNORECASE),     # "Before doing..."
    re.compile(r"^\s*despu[eé]s\s+de\s+\w", re.IGNORECASE),# "After doing..."
    re.compile(r"\bcomo\s+(hago|hacemos|debemos)\b", re.IGNORECASE),
    re.compile(r"\bpasos?\s+(para|de)\b", re.IGNORECASE),
    re.compile(r"\b(recuerda|asegur[aá]te)\s+de\b", re.IGNORECASE),
]


# World/regulatory patterns: durable external facts, code requirements.
# AE-domain examples:
#   "The NEC requires GFCI on outdoor receptacles."
#   "Chicago code mandates conduit for residential branch circuits."
#   "Aurora IL permit fee is $80 for service upgrades."
_WORLD_PATTERNS = [
    re.compile(r"\bNEC\s+(requires|mandates|allows|prohibits)\b", re.IGNORECASE),
    re.compile(r"\bcode\s+(requires|mandates|prohibits|states)\b", re.IGNORECASE),
    re.compile(r"\b(article|section)\s+\d+(\.\d+)?\b", re.IGNORECASE),  # NEC article refs
    re.compile(r"\b(must|shall)\s+(be|have|comply|conform)\b", re.IGNORECASE),
    re.compile(r"\bpermit\s+fee\s+is\b", re.IGNORECASE),
    re.compile(r"\b(regulation|statute|ordinance|standard)\s+(states|requires)\b", re.IGNORECASE),
]


# Mental-model patterns: summary, inference, conclusion language.
# AE-domain examples:
#   "We've concluded that Lennar prefers EOM invoicing over per-job."
#   "It seems Vibha's remodel timeline is driven by HOA approval."
_MENTAL_MODEL_PATTERNS = [
    re.compile(r"\b(we|i)('ve| have)?\s+concluded\b", re.IGNORECASE),
    re.compile(r"^\s*it\s+seems\b", re.IGNORECASE),
    re.compile(r"^\s*it\s+(appears|looks like)\b", re.IGNORECASE),
    re.compile(r"\b(in summary|to summarize|overall)\b", re.IGNORECASE),
    re.compile(r"\bthe pattern\s+(is|seems to be)\b", re.IGNORECASE),
    re.compile(r"\bwe('ve| have)?\s+(found|noticed|observed)\s+that\b", re.IGNORECASE),
    re.compile(r"^\s*(my|our)\s+(theory|hypothesis|read)\s+is\b", re.IGNORECASE),
]


def classify_memory_kind(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Classify free-text memory content into a memory kind.

    Args:
        text: the memory content to classify.
        metadata: optional metadata dict; if it carries an explicit "kind" key
            and the value is a recognized kind, that wins over heuristics.
            Used for callers that already know the kind (e.g., dream-engine
            insight generation).

    Returns:
        One of the strings in `memory_types.MEMORY_KINDS`. Default is
        "experience" when no pattern matches.
    """
    if metadata is not None:
        provided = metadata.get("kind")
        if isinstance(provided, str) and provided in MEMORY_KINDS:
            return provided

    if not text or not text.strip():
        return "unknown"

    if any(pat.search(text) for pat in _PROCEDURAL_PATTERNS):
        return "procedural"

    if any(pat.search(text) for pat in _WORLD_PATTERNS):
        return "world"

    if any(pat.search(text) for pat in _MENTAL_MODEL_PATTERNS):
        return "mental_model"

    return "experience"
