"""
dataset_v2 — Paraphrase ground-truth generator for the Neural Memory benchmark.

Why this exists: dataset.QueryGenerator builds queries by sampling 5 random
words from the source memory text. That is lexical leakage — recall@k is
trivially high because the query IS a bag-of-tokens from the target. The
embedding model can't help cheat that, but BM25/FTS does, and the resulting
metric tells you nothing about semantic understanding.

This module generates (statement, query) pairs where:

  * The statement asserts a unique fact about a synthetic anchor entity.
  * The query asks about the SAME anchor with DIFFERENT verbs/structure.
  * Ground truth is exactly one memory id.
  * The "answer" tokens (the part that resolves the query) appear only in
    the statement, NOT in the query.

The synthetic entity vocabulary is randomised so it cannot match anything in
an embedding model's pretraining corpus — recall has to come from semantic
proximity in the embedding model + the memory system's retrieval pipeline,
not from a memorised real-world fact or from a token overlap.

Output schema matches dataset.MasterDataset so existing suites can swap
generators with no other changes:

    {
        "id": "<unique str>",
        "text": "<the statement, ~60-180 chars>",
        "label": "paraphrase:<topic>",
        "metadata": {
            "topic": "...",
            "anchor": "<entity name>",
            "answer_tokens": ["..."],   # tokens that ONLY appear in text
            "type": "paraphrase",
        },
    }

Queries:

    {
        "query": "<paraphrased question>",
        "ground_truth_ids": ["<single id>"],
        "label": "paraphrase:<topic>",
        "anchor": "<entity name>",
        "leakage_score": <float 0..1>,   # measured token overlap query<->statement
    }
"""
from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, Iterable, List, Tuple


# Synthetic vocab designed to be GLOBALLY UNIQUE so no real-world prior can leak.
# Two-letter prefix + numeric suffix + random consonant cluster.
_SYLLABLES_HEAD = ["zr", "vl", "kr", "br", "tn", "qr", "xn", "wf", "pl", "dr"]
_SYLLABLES_TAIL = ["ax", "om", "uk", "ev", "ip", "ya", "ol", "ex", "uth", "ar"]


def _coined_word(rng: random.Random) -> str:
    """Coin a pronounceable but non-English token."""
    return f"{rng.choice(_SYLLABLES_HEAD)}{rng.choice('aeiou')}{rng.choice(_SYLLABLES_TAIL)}{rng.randint(10, 99)}"


def _normalise_tokens(text: str) -> List[str]:
    """Lowercase + alphanumeric tokens, stopword-stripped."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "which", "what",
    "when", "where", "how", "did", "does", "was", "were", "has", "have", "had",
    "are", "you", "your", "our", "their", "its", "into", "onto", "out",
    "about", "after", "before", "all", "any", "but", "not", "now", "on",
    "in", "of", "to", "is", "as", "an", "at", "by", "be",
})


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
# Each topic has:
#   * a STATEMENT template that uses {anchor} + {answer} (and maybe {extra}).
#   * 2-3 QUERY templates that share only {anchor} with the statement, never
#     {answer}. The verb/predicate is rephrased.
#
# The statement is what gets stored. The query is what gets searched.
# Ground truth: the statement is the one and only correct hit.

_TOPIC_BANK: List[Dict[str, Any]] = [
    {
        "topic": "ownership",
        "statement": "Component {anchor} is maintained by team {answer} as of {extra}.",
        "queries": [
            "Who owns {anchor}?",
            "Which team is responsible for {anchor}?",
            "{anchor} maintainer.",
        ],
        "answer_pool": [
            "platform-core", "ingest-pipeline", "graph-runtime", "edge-cache",
            "rotor-eng", "perception", "schema-stewards", "feature-foundry",
            "billing-rails", "observability", "trust-and-safety", "narrative-tools",
        ],
        "extra_pool": [
            "Q1 2026", "the latest reorg", "the v3 charter", "the migration kickoff",
        ],
    },
    {
        "topic": "incident",
        "statement": "Incident on {anchor} was triggered by {answer} during the rollout window.",
        "queries": [
            "What caused the {anchor} outage?",
            "Why did {anchor} fail?",
            "{anchor} incident root cause.",
        ],
        "answer_pool": [
            "a misconfigured retry budget", "an expired TLS bundle",
            "a stale feature flag", "an unbounded fan-out", "a partial schema migration",
            "a leaked goroutine pool", "a clock-skew on the leader", "an OOM in the writer",
        ],
        "extra_pool": ["the audit", "the postmortem"],
    },
    {
        "topic": "deprecation",
        "statement": "The {anchor} interface is being replaced with {answer} starting next quarter.",
        "queries": [
            "What will replace {anchor}?",
            "{anchor} successor.",
            "Migration target for {anchor}.",
        ],
        "answer_pool": [
            "a streaming consumer", "a typed SDK shim", "a sidecar gateway",
            "an event-sourced view", "a columnar replacement", "a CRDT log",
            "a managed broker", "an inline edge function",
        ],
        "extra_pool": [],
    },
    {
        "topic": "metric",
        "statement": "Latency of {anchor} is bounded by {answer} under steady state.",
        "queries": [
            "How fast is {anchor}?",
            "{anchor} latency budget.",
            "Performance ceiling of {anchor}.",
        ],
        "answer_pool": [
            "12 ms p95", "47 ms p99", "180 microseconds", "7 ms median",
            "3 seconds end-to-end", "240 ms cold-start", "55 ms warm-start",
        ],
        "extra_pool": [],
    },
    {
        "topic": "config",
        "statement": "Configuration knob {anchor} defaults to {answer} on the production tier.",
        "queries": [
            "Default value of {anchor}?",
            "{anchor} production default.",
            "Out-of-the-box setting for {anchor}.",
        ],
        "answer_pool": [
            "0.85", "true", "false", "120", "auto", "exponential",
            "round-robin", "least-loaded", "weighted-random", "deterministic",
        ],
        "extra_pool": [],
    },
    {
        "topic": "decision",
        "statement": "We chose {answer} as the persistence backend for {anchor} after the spike.",
        "queries": [
            "Which backend is {anchor} on?",
            "Storage choice for {anchor}.",
            "What persists {anchor}?",
        ],
        "answer_pool": [
            "an embedded KV store", "a columnar DB", "the existing relational tier",
            "a custom log-structured engine", "an in-memory tier with snapshots",
            "the compliance-grade ledger",
        ],
        "extra_pool": [],
    },
    {
        "topic": "person",
        "statement": "Approval authority for {anchor} sits with {answer}.",
        "queries": [
            "Who signs off on {anchor}?",
            "{anchor} approver.",
            "Who do I escalate {anchor} to?",
        ],
        "answer_pool": [
            "the deputy director", "the on-call principal",
            "the architecture review board", "the compliance liaison",
            "the rotation chair", "the platform chief",
        ],
        "extra_pool": [],
    },
    {
        "topic": "schedule",
        "statement": "Maintenance for {anchor} runs every {answer} during the quiet window.",
        "queries": [
            "When is {anchor} maintained?",
            "{anchor} maintenance cadence.",
            "How often is {anchor} serviced?",
        ],
        "answer_pool": [
            "second Tuesday", "third weekend", "last Friday of the month",
            "alternate Wednesday", "first business day after payroll",
        ],
        "extra_pool": [],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ParaphraseGenerator:
    """Yields (memory, query) pairs with disjoint-ish vocabulary.

    Anchors are coined nonsense tokens — they cannot collide with the
    embedding model's pretraining and they cannot collide between memories.
    That gives a clean 1:1 ground truth: each anchor is mentioned in exactly
    one statement, so the only "right" hit for a query is that statement.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._used_anchors: set[str] = set()

    def _fresh_anchor(self) -> str:
        for _ in range(100):
            tok = _coined_word(self._rng)
            if tok not in self._used_anchors:
                self._used_anchors.add(tok)
                return tok
        # Pathological fallback — append entropy
        tok = _coined_word(self._rng) + "".join(self._rng.choices(string.ascii_lowercase, k=4))
        self._used_anchors.add(tok)
        return tok

    def _paraphrase_pair(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        topic = self._rng.choice(_TOPIC_BANK)
        anchor = self._fresh_anchor()
        answer = self._rng.choice(topic["answer_pool"])
        extra = self._rng.choice(topic["extra_pool"]) if topic["extra_pool"] else ""

        # Some statements use {extra}, some don't — handle both gracefully.
        try:
            statement = topic["statement"].format(anchor=anchor, answer=answer, extra=extra)
        except KeyError:
            statement = topic["statement"].format(anchor=anchor, answer=answer)
        # Strip a trailing "during the rollout window." style fragment if {extra}
        # was empty and produced "as of ." or similar.
        statement = re.sub(r"\s+as of\s*\.\s*$", ".", statement)
        statement = re.sub(r"\s+during the\s*\.\s*$", ".", statement)

        query_template = self._rng.choice(topic["queries"])
        query = query_template.format(anchor=anchor)

        # Compute leakage as Jaccard between informative tokens.
        # Drop the anchor itself from both sides — it MUST appear in both,
        # that's how the question is grounded; the leakage we care about is
        # whether the query echoes the *answer* tokens.
        s_toks = set(_normalise_tokens(statement)) - {anchor}
        q_toks = set(_normalise_tokens(query)) - {anchor}
        if s_toks or q_toks:
            leak = len(s_toks & q_toks) / max(1, len(s_toks | q_toks))
        else:
            leak = 0.0

        mem_id = f"para-{topic['topic']}-{anchor}"
        memory = {
            "id": mem_id,
            "text": statement,
            "label": f"paraphrase:{topic['topic']}",
            "metadata": {
                "type": "paraphrase",
                "topic": topic["topic"],
                "anchor": anchor,
                "answer": answer,
                "answer_tokens": _normalise_tokens(answer),
            },
        }
        query_record = {
            "query": query,
            "ground_truth_ids": [mem_id],
            "label": f"paraphrase:{topic['topic']}",
            "anchor": anchor,
            "leakage_score": round(leak, 3),
        }
        return memory, query_record

    def generate(self, count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        memories: List[Dict[str, Any]] = []
        queries: List[Dict[str, Any]] = []
        for _ in range(count):
            mem, q = self._paraphrase_pair()
            memories.append(mem)
            queries.append(q)
        return memories, queries


# ---------------------------------------------------------------------------
# Cross-session continuity dataset
# ---------------------------------------------------------------------------

def generate_continuity_pairs(seed: int = 42, count: int = 50) -> List[Dict[str, Any]]:
    """Generate facts that will be stored in 'session 1' and queried in 'session N'.

    Each fact uses a unique anchor + answer. The 'session 1' write happens
    early, then noise sessions store unrelated content (also paraphrase-style
    so retrieval has to discriminate by meaning, not recency). The query in
    'session N' asks for the original fact's answer.
    """
    gen = ParaphraseGenerator(seed=seed)
    pairs: List[Dict[str, Any]] = []
    for _ in range(count):
        mem, q = gen._paraphrase_pair()
        pairs.append({
            "memory": mem,
            "query": q["query"],
            "expected_id": mem["id"],
            "expected_answer": mem["metadata"]["answer"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Conflict-quality dataset
# ---------------------------------------------------------------------------

_CONFLICT_PAIRS_TEMPLATE: List[Tuple[str, str, str]] = [
    # (anchor seed, original, replacement) — both share anchor, only ONE
    # should win after supersession. Both are stored under the same label so
    # the system has to detect the conflict semantically.
    ("ownership", "Component {a} is maintained by team alpha-cell.",
                  "Component {a} is maintained by team beta-loop after the reorg."),
    ("metric", "Latency of {a} is bounded by 50 ms p95 under load.",
               "Latency of {a} is bounded by 12 ms p95 after the rewrite."),
    ("config", "Default value of {a} is exponential.",
               "Default value of {a} is round-robin starting next release."),
    ("decision", "Backend for {a} is the relational tier.",
                 "Backend for {a} switched to the columnar engine."),
    ("schedule", "Maintenance of {a} runs every second Tuesday.",
                 "Maintenance of {a} now runs the third weekend."),
]


def generate_conflict_pairs(seed: int = 42, count: int = 30) -> List[Dict[str, Any]]:
    """Generate conflict triplets: (original, replacement, query, winner_text).

    The winner is always the REPLACEMENT (latest write). A correct system
    should rank the replacement above the original after both are stored.
    Both share an anchor so semantic retrieval finds both candidates.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    pgen = ParaphraseGenerator(seed=seed + 1)
    while len(out) < count:
        a = pgen._fresh_anchor()
        topic, original_tmpl, replacement_tmpl = rng.choice(_CONFLICT_PAIRS_TEMPLATE)
        original = original_tmpl.format(a=a)
        replacement = replacement_tmpl.format(a=a)

        query_templates = {
            "ownership": ["Who owns {a}?", "{a} maintainer."],
            "metric": ["How fast is {a}?", "{a} latency budget."],
            "config": ["Default value of {a}?", "{a} default setting."],
            "decision": ["Which backend powers {a}?", "{a} storage choice."],
            "schedule": ["When is {a} maintained?", "{a} maintenance cadence."],
        }[topic]
        q = rng.choice(query_templates).format(a=a)

        out.append({
            "anchor": a,
            "topic": topic,
            "original": original,
            "replacement": replacement,
            "query": q,
            "winner_text": replacement,
        })
    return out
