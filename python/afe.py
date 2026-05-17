"""Atomic Fact Extraction (AFE) for Mazemaker.

Splits long boilerplate-heavy memory turns into atomic facts that
recall can pinpoint directly, instead of returning a 5000-char wall
where the relevant 30-char fact is buried.

Three-stage extractor, LLM-free in the default path:

  Stage A:  Markdown structural extractor — pure regex (~5 ms/turn).
            Handles 80-95% of assistant-formatted content with
            headers/bullets/key:value patterns.

  Stage B:  spaCy NER fallback — entity-driven extraction (~50 ms/turn).
            Activated when Stage A yields zero facts.
            Requires `spacy` + `en_core_web_sm` (40 MB model).

  Stage C:  Tiny LLM fallback — optional, opt-in via env var.
            Default model: DeepHermes-3-Llama-3-3B-Preview on GPU,
            smollm2:1.7b on CPU.

Each extracted fact is a dict with keys:
  text     : 60-300 char atomic fact string (the body stored as a new memory)
  entity   : the entity (str or None)
  predicate: relation/field (str or None)
  value    : the value (str or None)
  source_span : (start, end) char offsets in original content
  stage    : "A" | "B" | "C"

The dream engine writes each fact as a new memory linked back to the
source via a 'supports' connection edge.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional


# ──────────────────────────────────────────────────────────────────────
#  Stage A — Markdown structural extractor
# ──────────────────────────────────────────────────────────────────────

# Match markdown headers: # H1, ## H2, ### H3, #### H4
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Match bulleted "**Field**: value" pattern (most common in assistant output)
# Captures predicate + value
_BULLET_KV_RE = re.compile(
    r"^[\-\*\+]\s+\*\*([^*]+?)\*\*\s*:\s*(.+?)\s*$",
    re.MULTILINE,
)

# Bullet without bold: "- Field: value"
# Widened from {1,30}? to {1,60}? — the 30-char cap silently rejected
# legitimate multi-word predicates ("Estimated annual revenue",
# "Total cumulative paid out") and dropped those bullets from AFE.
_BULLET_KV_PLAIN_RE = re.compile(
    r"^[\-\*\+]\s+([A-Z][A-Za-z][A-Za-z\s]{1,60}?)\s*:\s+(.+?)\s*$",
    re.MULTILINE,
)

# Match bold entity headers like "#### **Bay Medical Sacred Heart**"
_BOLD_ENTITY_HEADER_RE = re.compile(
    r"^#{2,6}\s+\*\*([^*]+?)\*\*\s*$",
    re.MULTILINE,
)

# Match inline "**entity** (attribute)" or "**entity**: brief"
_INLINE_BOLD_ASSIGN_RE = re.compile(
    r"\*\*([^*]+?)\*\*\s*[:\-]\s+([^.;\n]+)",
)

# Detect dollar amounts, percentages, version numbers, phone numbers
_NUMERIC_TOKEN_RE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?[KkMmBb]?"             # $1,500 / $70 / $2.5K
    r"|\d+(?:\.\d+)?\s*(?:%|°|GB|MB|KB|TB|kg|km|m|cm|mph|kHz|MHz|GHz)"
    r"|\d+(?:\.\d+)?\s+(?:inches|inch|feet|ft|hrs?|mins?|seconds?)"
    r"|\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"      # phone (850) 718-2000
    r"|\b\d+(?:\.\d+)+\b"                        # version 1.26.0
    r"|\b\d{4}[\-/]\d{1,2}[\-/]\d{1,2}\b"        # 2024-09-15
)


def _looks_atomic(s: str) -> bool:
    """Reject noise: empty, too-short, too-long, or pure prose."""
    if not s:
        return False
    s = s.strip()
    if len(s) < 4 or len(s) > 280:
        return False
    # Reject pure prose: no numbers/proper-nouns/structured token
    if not _NUMERIC_TOKEN_RE.search(s) and not re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", s):
        return False
    return True


def _stage_a_markdown(content: str) -> List[Dict[str, Any]]:
    """Extract facts from markdown-structured content."""
    facts: List[Dict[str, Any]] = []

    # Track the most recent bold-entity header so bullet KVs can be
    # attributed to "Entity, predicate: value" instead of just "predicate: value".
    current_entity: Optional[str] = None

    # First pass: collect all headers so we know section boundaries.
    headers = list(_HEADER_RE.finditer(content))

    # Find bold-entity headers (e.g. "#### **Bay Medical Sacred Heart**")
    bold_entities = list(_BOLD_ENTITY_HEADER_RE.finditer(content))

    # Walk content line by line, tracking the most recent entity header.
    def entity_at(pos: int) -> Optional[str]:
        last: Optional[str] = None
        for m in bold_entities:
            if m.start() <= pos:
                last = m.group(1).strip()
            else:
                break
        return last

    # Extract bulleted key-value pairs
    for m in _BULLET_KV_RE.finditer(content):
        predicate = m.group(1).strip()
        value = m.group(2).strip()
        entity = entity_at(m.start())
        # Compose atomic fact text
        if entity:
            text = f"{entity}, {predicate}: {value}"
        else:
            text = f"{predicate}: {value}"
        if _looks_atomic(text):
            facts.append({
                "text": text,
                "entity": entity,
                "predicate": predicate,
                "value": value,
                "source_span": (m.start(), m.end()),
                "stage": "A",
            })

    # Plain "Field: value" bullets (no bold)
    for m in _BULLET_KV_PLAIN_RE.finditer(content):
        predicate = m.group(1).strip()
        value = m.group(2).strip()
        # Skip if the predicate is generic prose word ("Here", "Note", etc.)
        if predicate.lower() in {"note", "here", "for example", "tip", "warning", "important"}:
            continue
        entity = entity_at(m.start())
        if entity:
            text = f"{entity}, {predicate}: {value}"
        else:
            text = f"{predicate}: {value}"
        if _looks_atomic(text):
            facts.append({
                "text": text,
                "entity": entity,
                "predicate": predicate,
                "value": value,
                "source_span": (m.start(), m.end()),
                "stage": "A",
            })

    # Inline bold-entity assignments: "**Bay Medical Sacred Heart**: hospital"
    for m in _INLINE_BOLD_ASSIGN_RE.finditer(content):
        entity = m.group(1).strip()
        value = m.group(2).strip()
        # Skip if already captured via bullet
        if any(f.get("entity") == entity for f in facts):
            continue
        text = f"{entity}: {value}"
        if _looks_atomic(text):
            facts.append({
                "text": text,
                "entity": entity,
                "predicate": None,
                "value": value,
                "source_span": (m.start(), m.end()),
                "stage": "A",
            })

    # Deduplicate by text (case-insensitive)
    seen = set()
    out = []
    for f in facts:
        k = f["text"].lower().strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Stage B — spaCy NER fallback (optional)
# ──────────────────────────────────────────────────────────────────────

_spacy_nlp = None


def _load_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_nlp = False  # sentinel — don't retry
    return _spacy_nlp


def _stage_b_ner(content: str) -> List[Dict[str, Any]]:
    """Extract entity+value facts via spaCy NER. Returns [] if spaCy missing."""
    nlp = _load_spacy()
    if not nlp:
        return []
    facts: List[Dict[str, Any]] = []
    doc = nlp(content[:50_000])  # safety cap for very long content
    # Group entities by sentence for context.
    interesting_labels = {"PERSON", "ORG", "MONEY", "DATE", "QUANTITY",
                          "PERCENT", "GPE", "FAC", "CARDINAL"}
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in interesting_labels]
        if len(ents) >= 2:
            # Form "<entity1> ... <entity2>" snippet — atomic fact pairing.
            text = sent.text.strip()
            if _looks_atomic(text):
                facts.append({
                    "text": text[:280],
                    "entity": ents[0].text,
                    "predicate": ents[0].label_,
                    "value": ", ".join(e.text for e in ents[1:]),
                    "source_span": (sent.start_char, sent.end_char),
                    "stage": "B",
                })
    return facts


# ──────────────────────────────────────────────────────────────────────
#  Stage C — Tiny-LLM fallback (opt-in)
# ──────────────────────────────────────────────────────────────────────

def _stage_c_llm(content: str, model: str) -> List[Dict[str, Any]]:
    """Extract facts via a local tiny LLM (default DeepHermes-3-3B-Preview).

    Activated only via MAZEMAKER_AFE_LLM_FALLBACK=1 + Stage A/B yielded nothing.
    Talks to ollama by default (Mazemaker config doesn't ship its own llm).
    """
    try:
        import subprocess
        # Hard truncation at 3000 chars: the local tiny-LLM (default
        # DeepHermes-3-3B) has an 8k context budget; the prompt header +
        # the response we're asking for already eat ~700 tokens. Bigger
        # windows return malformed JSON in practice. Log when we drop
        # tail content so an operator inspecting low-recall facts can
        # see the cause — the previous silent truncation made it look
        # like AFE was hallucinating from data that wasn't there.
        TRUNCATE_AT = 3000
        if len(content) > TRUNCATE_AT:
            logger.info(
                "AFE Stage C: truncating %d-char source to %d for LLM "
                "(facts may miss tail content)",
                len(content), TRUNCATE_AT,
            )
        prompt = (
            "Extract specific atomic facts from the passage below as a JSON "
            "list of strings. Include both:\n"
            "  (1) User-side statements: preferences, opinions, decisions, "
            "intentions, personal details (job, location, possessions, "
            "relationships, habits) that the USER expresses about themselves. "
            "Phrase as 'user prefers X', 'user is a Y', 'user owns Z', etc.\n"
            "  (2) Concrete entity facts: specific numbers, dollar amounts, "
            "named entities, dates, addresses, version identifiers.\n\n"
            "Each fact must be a short atomic string under 200 characters. "
            "Skip generic advice and boilerplate. Skip facts about the "
            "assistant. Output ONLY the JSON array, nothing else.\n\n"
            f"PASSAGE:\n{content[:TRUNCATE_AT]}\n\nFACTS:"
        )
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return []
        import json
        out = result.stdout.strip()
        # Strip code fences
        out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.MULTILINE).strip()
        # Find first JSON array
        m = re.search(r"\[.*\]", out, re.DOTALL)
        if not m:
            return []
        facts_raw = json.loads(m.group(0))
        facts: List[Dict[str, Any]] = []
        for f in facts_raw:
            if isinstance(f, str) and _looks_atomic(f):
                facts.append({
                    "text": f.strip()[:280],
                    "entity": None,
                    "predicate": None,
                    "value": None,
                    "source_span": (0, len(content)),
                    "stage": "C",
                })
            elif isinstance(f, dict) and "text" in f and _looks_atomic(f["text"]):
                facts.append({
                    "text": f["text"][:280],
                    "entity": f.get("entity"),
                    "predicate": f.get("predicate"),
                    "value": f.get("value"),
                    "source_span": (0, len(content)),
                    "stage": "C",
                })
        return facts
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def extract_atomic_facts(
    content: str,
    *,
    min_content_length: int = 500,
    enable_a: bool = True,
    enable_ner: bool = True,
    enable_llm_fallback: bool = False,
    llm_model: str = "DeepHermes-3-Llama-3-3B-Preview",
) -> List[Dict[str, Any]]:
    """Extract atomic facts from a memory content string.

    Returns a list of fact dicts. Empty list if content is too short or
    no extractable atomic facts found.

    Args:
        content:          The memory content (typically a long assistant turn).
        min_content_length: Skip content shorter than this — short content
                          is already atomic.
        enable_a:         Run Stage A (markdown bullet extraction). Default
                          True. Disable to force fall-through to NER/LLM
                          on corpora where bullets are present but the
                          useful facts are in surrounding prose.
        enable_ner:       Run Stage B (spaCy NER) if Stage A returns nothing.
        enable_llm_fallback: Run Stage C (tiny LLM) if A+B return nothing.
                          Default False to keep Mazemaker LLM-free.
        llm_model:        Ollama model name for Stage C.
    """
    if not content or len(content) < min_content_length:
        return []
    facts: List[Dict[str, Any]] = []
    if enable_a:
        facts = _stage_a_markdown(content)
    if not facts and enable_ner:
        facts = _stage_b_ner(content)
    if not facts and enable_llm_fallback:
        facts = _stage_c_llm(content, llm_model)
    return facts
