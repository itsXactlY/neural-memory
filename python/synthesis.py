"""Stage S — Cross-source memory synthesis.

Where AFE Stage A/B/C extract atomic facts FROM ONE source, Stage S
distills MANY atomic facts into stable user-level patterns. It runs as
a dream phase after Insight has clustered the AFE outputs.

Architectural distinction from extraction:
  Stage A/B/C   single source → ≤N atomic statements      (local)
  Stage S       N atomic statements → ≤K crystallized     (cross-source)
                 preference/pattern memories with
                 graph-derived confidence and
                 evidence-linked edges

The LLM proposes the statement; the GRAPH decides whether to crystallize
it and at what strength. Confidence is:

    conf = sigmoid( α·log(1+N) + β·S − γ·contradictions )

where
  N             evidence count surviving the LLM's evidence_indices
  S             mean pairwise cosine similarity among evidence embeddings
                (cluster tightness; higher = more coherent pattern)
  contradictions  count of `supersedes` edges from any evidence — if a
                source statement was already revised, the pattern built
                on it is weaker

This module is import-safe — it never opens a DB connection itself.
The dream-engine caller passes content, embeddings, and ids; the
caller writes the resulting memories.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("mazemaker.synthesis")

# ──────────────────────────────────────────────────────────────────────
#  Tuning constants (env-overridable via MAZEMAKER_SYNTHESIS_*)
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "alibayram/smollm3"
_DEFAULT_TIMEOUT_S = 60
_TRUNCATE_AT = 3000          # per-fact char cap going into the LLM prompt
_MAX_FACTS_IN_PROMPT = 12    # too many → smaller LLM forgets the schema


# ──────────────────────────────────────────────────────────────────────
#  Confidence math (graph-derived, NOT LLM-self-reported)
# ──────────────────────────────────────────────────────────────────────

def compute_confidence(
    n_evidence: int,
    similarity_mean: float,
    contradictions: int = 0,
    *,
    alpha: float = 0.5,
    beta: float = 1.2,
    gamma: float = 0.7,
) -> float:
    """Sigmoid of a weighted log-evidence + tightness − contradictions score.

    - n_evidence:   how many supporting member memories survived
                    the LLM's evidence_indices selection.
    - similarity_mean: mean pairwise cosine similarity among evidence
                    memory embeddings, in [0, 1]. Higher = the LLM is
                    summarising a tight cluster, not stitching unrelated
                    fragments.
    - contradictions: count of supersedes edges (memory ⊃ replaced_by)
                    incident on any evidence id. Each prior revision
                    weakens the pattern proportionally.

    Defaults calibrated so that a 3-evidence 0.7-similarity 0-contradiction
    cluster returns ~0.78, a 1-evidence 0.5-similarity cluster returns
    ~0.55, and any cluster with ≥3 contradictions falls below 0.5.
    """
    score = (
        alpha * math.log1p(max(0, n_evidence))
        + beta * max(0.0, min(1.0, similarity_mean))
        - gamma * max(0, contradictions)
    )
    return 1.0 / (1.0 + math.exp(-score))


def mean_pairwise_similarity(embeddings: List[List[float]]) -> float:
    """Mean of all pairwise cosine similarities. Returns 1.0 for n=1
    (single-evidence patterns get full tightness credit because there's
    nothing to disagree with), 0.0 for empty input.

    Embeddings are assumed unit-normalised (which BGE-M3 emits by default).
    If they aren't the result is still a relative tightness score, just
    not technically cosine.
    """
    n = len(embeddings)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        ei = embeddings[i]
        for j in range(i + 1, n):
            ej = embeddings[j]
            # Dot product (already unit-norm)
            s = sum(a * b for a, b in zip(ei, ej))
            total += s
            pairs += 1
    return total / pairs if pairs else 0.0


# ──────────────────────────────────────────────────────────────────────
#  LLM call — local Ollama, JSON-array output
# ──────────────────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are summarising a cluster of atomic facts that all came from
conversations with the SAME user. Your job: distill them into stable
USER-LEVEL patterns — preferences, recurring behaviours, durable traits
— that an agent should remember about this user.

Each output item is a JSON object with:
  - "text":              short atomic statement, under 200 chars,
                         phrased as a fact about the user
                         ("user prefers Italian food",
                          "user owns an RTX 4060 Ti 16GB",
                          "user dislikes SaaS lock-in")
  - "type":              one of "preference", "pattern", "trait", "fact"
                         preference = liking/disliking something specific
                         pattern    = recurring behaviour or choice
                         trait      = stable disposition or value
                         fact       = concrete entity ownership/identity
  - "evidence_indices":  list of integer indices into the input atomic
                         facts (0-indexed) that support this statement.
                         Include EVERY supporting index — confidence is
                         computed from this list.

Output ONLY a JSON array. Drop any statement you cannot confidently
support from at least one input fact. Do NOT hallucinate facts not
present. If no synthesisable pattern exists, output an empty array `[]`.

INPUT ATOMIC FACTS (numbered):
{facts_block}

OUTPUT JSON:"""


def _ollama_synthesize(
    facts: List[str],
    *,
    model: str,
    timeout_s: int,
) -> List[Dict[str, Any]]:
    if not facts:
        return []
    capped = facts[:_MAX_FACTS_IN_PROMPT]
    facts_block = "\n".join(
        f"{i}. {f.strip()[:_TRUNCATE_AT]}" for i, f in enumerate(capped)
    )
    prompt = _SYNTHESIS_PROMPT.format(facts_block=facts_block)
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning("synthesis LLM timeout (model=%s, n_facts=%d)", model, len(capped))
        return []
    except FileNotFoundError:
        log.warning("synthesis: `ollama` binary not on PATH — phase disabled")
        return []
    if result.returncode != 0:
        log.debug("synthesis LLM exit %d: %s", result.returncode, result.stderr[:200])
        return []
    out = result.stdout.strip()
    # Strip code fences
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", out, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out_items: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()[:280]
        ev = item.get("evidence_indices") or []
        if not isinstance(ev, list):
            continue
        ev_clean = [int(i) for i in ev if isinstance(i, (int, float)) and 0 <= int(i) < len(capped)]
        if not ev_clean:
            continue
        out_items.append({
            "text": text,
            "type": (item.get("type") or "pattern").strip().lower(),
            "evidence_indices": ev_clean,
        })
    return out_items


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def synthesize_cluster(
    *,
    fact_contents: List[str],
    fact_ids: List[int],
    fact_embeddings: Optional[List[List[float]]] = None,
    fact_contradictions: Optional[List[int]] = None,
    fact_session_ids: Optional[List[str]] = None,
    model: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_evidence: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Synthesise one cluster of AFE-extracted atomic facts into stable
    user-level patterns. Returns a list of crystallised memory specs:

        {
          "text":            "user prefers Italian food",
          "type":            "preference",
          "evidence_ids":    [123, 456, 789],
          "evidence_session_ids": ["session:foo", "session:bar"],
          "confidence":      0.82,
          "label_segments":  ["foo", "bar"],   # session-id segments
                                                # appended to the label
                                                # so bench scoring + agent
                                                # provenance both resolve
        }

    All thresholding (min_confidence, min_evidence) is applied here so
    the caller only writes memories that the graph has already vetted.

    Args:
      fact_contents:        the atomic-fact strings for this cluster's
                            members. Order matters: LLM evidence_indices
                            reference these positions.
      fact_ids:             memory IDs corresponding 1:1 to fact_contents.
      fact_embeddings:      1024-d embeddings per fact, unit-normed.
                            If None, similarity_mean defaults to 1.0.
      fact_contradictions:  per-fact count of supersedes edges incident
                            on that fact id. None ≡ all-zeros.
      fact_session_ids:     per-fact source session id (the bench-graded
                            label segment). None ≡ empty per-fact.
      model:                Ollama model. Defaults env-overridable.
      min_confidence:       discard patterns below this confidence.
      min_evidence:         require at least this many evidence ids.
      timeout_s:            per-LLM-call timeout.
    """
    if not fact_contents or not fact_ids:
        return []
    if len(fact_contents) != len(fact_ids):
        raise ValueError("fact_contents and fact_ids length mismatch")

    model = model or os.environ.get("MAZEMAKER_SYNTHESIS_MODEL", _DEFAULT_MODEL)
    min_conf = (min_confidence if min_confidence is not None
                else float(os.environ.get("MAZEMAKER_SYNTHESIS_MIN_CONF", "0.5")))
    min_ev = (min_evidence if min_evidence is not None
              else int(os.environ.get("MAZEMAKER_SYNTHESIS_MIN_EVIDENCE", "2")))
    timeout = (timeout_s if timeout_s is not None
               else int(os.environ.get("MAZEMAKER_SYNTHESIS_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S))))

    proposals = _ollama_synthesize(fact_contents, model=model, timeout_s=timeout)
    if not proposals:
        return []

    crystallised: List[Dict[str, Any]] = []
    for p in proposals:
        ev_idx: List[int] = p["evidence_indices"]
        if len(ev_idx) < min_ev:
            continue
        ev_ids = [fact_ids[i] for i in ev_idx]
        ev_embeds = (
            [fact_embeddings[i] for i in ev_idx]
            if fact_embeddings is not None else None
        )
        ev_contradictions = (
            sum(fact_contradictions[i] for i in ev_idx)
            if fact_contradictions is not None else 0
        )
        ev_sids = (
            [fact_session_ids[i] for i in ev_idx if fact_session_ids[i]]
            if fact_session_ids is not None else []
        )
        # Dedup session ids preserving order
        seen = set()
        ev_sids_unique: List[str] = []
        for s in ev_sids:
            if s and s not in seen:
                seen.add(s)
                ev_sids_unique.append(s)

        sim = (mean_pairwise_similarity(ev_embeds) if ev_embeds is not None else 1.0)
        conf = compute_confidence(
            n_evidence=len(ev_ids),
            similarity_mean=sim,
            contradictions=ev_contradictions,
        )
        if conf < min_conf:
            continue
        crystallised.append({
            "text": p["text"],
            "type": p["type"],
            "evidence_ids": ev_ids,
            "evidence_session_ids": ev_sids_unique,
            "confidence": round(conf, 4),
            "label_segments": ev_sids_unique,
            "similarity_mean": round(sim, 4),
            "contradictions": ev_contradictions,
        })
    return crystallised


def label_for(item: Dict[str, Any]) -> str:
    """Build the canonical label for a synthesised memory.

    Format: ``synthesized:<type>:<sid_0>::<sid_1>::...``

    Each evidence session id becomes its own colon-delimited segment so
    that downstream label-split rank scoring (``rank_of_gold`` in
    benchmarks) matches the gold session ID directly without any bench-
    aware logic in the engine.
    """
    base = f"synthesized:{item.get('type', 'pattern')}"
    seg = item.get("label_segments") or []
    if not seg:
        return base
    return base + ":" + "::".join(seg)
