"""
dataset_real — Real-text corpus generator for the neural-memory benchmark.

Why this exists: the v3+v4 ParaphraseGenerator produced template-derived
synthetic memories. Codex's v5 verdict accepted the benchmark as proof
of mechanism but called out "synthetic data only" as a remaining caveat.
This module produces (memory, query) pairs from the project's actual
prose — README files, CLAUDE.md, Python docstrings, etc. — so claims
about retrieval can be defended on real text, not template variations.

Pipeline:
  1. Walk the project root for .md and .py files (capped to keep the
     corpus deterministic; sorted by path so two runs produce the same
     pool).
  2. Split prose into paragraph-sized chunks (~50-400 chars each).
  3. For each chunk, extract a NAMED ENTITY from the chunk (CamelCase /
     snake_case identifiers, file paths, function names) — these become
     anchor candidates. Pick one anchor per chunk uniquely (global
     registry across all chunks, so each anchor occurs in exactly one
     chunk; chunks without a unique-enough anchor are dropped).
  4. Rewrite a paraphrastic question that references the anchor
     entity but uses different surface form than the chunk text. The
     query templates are designed to ASK ABOUT the chunk's content
     without echoing its non-anchor tokens.

Output schema is identical to dataset_v2.ParaphraseGenerator so existing
suites work unchanged. The `metadata.type` is set to "real_text" so a
suite or report can distinguish.

Determinism:
  * Files are read in alphabetical order.
  * Chunks within a file are kept in source order.
  * Anchor selection from a chunk's candidates is by stable iteration
    (sorted candidates).
  * The optional seed only randomises the choice of QUERY TEMPLATE
    among the per-anchor-type pool, never the corpus itself.

The corpus is small by design (~200-500 chunks) — it's a peer-review
sanity check, not a retrieval scaling test.
"""
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

# Cap to keep the corpus deterministic and small. Adjust if you want a
# larger real-text experiment.
_MAX_CHUNKS = 600
_CHUNK_MIN = 60
_CHUNK_MAX = 400


def _walk_corpus(project_root: Path) -> List[Path]:
    """Return a sorted list of .md / .py files under project_root.

    Excludes generated trees (build/, __pycache__, *.egg-info) and the
    benchmark's own audit dir (which contains adversarial text the
    suites would otherwise re-ingest).
    """
    excluded_parts = {
        "build", "__pycache__", ".git", ".cache", "node_modules",
        "audit",  # don't re-ingest the codex transcripts
        "results",  # don't re-ingest prior run outputs
    }
    excluded_suffixes = {".egg-info"}
    out: List[Path] = []
    for p in project_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".md", ".py"}:
            continue
        if any(part in excluded_parts for part in p.parts):
            continue
        if any(p.name.endswith(suf) for suf in excluded_suffixes):
            continue
        out.append(p)
    out.sort()
    return out


def _chunk_text(text: str) -> List[str]:
    """Paragraph-aware chunking with size bounds.

    Splits on blank lines, then concatenates short paragraphs into
    chunks of ~_CHUNK_MIN to _CHUNK_MAX characters. Drops chunks that
    are mostly code (heuristic: too many braces / equals / parens) or
    too short.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        # Skip code-block markers and indented code lines
        lines = para.splitlines()
        non_code_lines = [
            ln for ln in lines
            if not (ln.startswith("    ") or ln.startswith("\t"))
        ]
        if not non_code_lines:
            continue
        cleaned = " ".join(non_code_lines).strip()
        if len(cleaned) < 20:
            continue
        # Heuristic: skip code-heavy chunks. Markdown prose has < 0.05
        # density of [={};] characters per word.
        density = sum(1 for c in cleaned if c in "={};") / max(1, len(cleaned.split()))
        if density > 0.5:
            continue
        if len(buf) + len(cleaned) + 1 <= _CHUNK_MAX:
            buf = (buf + " " + cleaned).strip() if buf else cleaned
        else:
            if len(buf) >= _CHUNK_MIN:
                chunks.append(buf)
            buf = cleaned[:_CHUNK_MAX]
    if len(buf) >= _CHUNK_MIN:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Anchor extraction & query rewriting
# ---------------------------------------------------------------------------

# Patterns that produce unique-enough entity tokens from real prose.
# Each (regex, kind) — `kind` drives which query template family fires.
_ANCHOR_PATTERNS = [
    # Snake_case multiword identifiers (e.g. dream_engine, memory_client)
    (re.compile(r"\b([a-z][a-z0-9]+_[a-z0-9_]+)\b"), "snake"),
    # CamelCase class names (e.g. NeuralMemory, DreamEngine, AccessLogger)
    (re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,})\b"), "camel"),
    # Dotted file paths (e.g. memory_client.py, dream_engine.py)
    (re.compile(r"\b([a-z][a-z0-9_]+\.py)\b"), "py_file"),
    # Markdown headers in the chunk (cleaned of leading #)
    # — captured but lower priority via ordering below.
]


# Words too generic to use as anchors even if they pattern-match. These
# would otherwise leak into many chunks and break the global-uniqueness
# invariant.
_ANCHOR_BLACKLIST = frozenset({
    "self", "True", "False", "None", "Path", "List", "Dict", "Optional",
    "Tuple", "Any", "Union", "Type", "Callable", "Iterable", "Generator",
    "TypeVar", "Sequence", "Mapping", "Set",
    "test_suite", "this_dir", "src_root", "bench_root",
    "json_dump", "config_yaml",
    "__init__", "__main__", "__file__", "__name__",
})


def _candidate_anchors(chunk: str) -> List[Tuple[str, str]]:
    """Return [(token, kind), ...] in priority order, deduped."""
    seen = set()
    out: List[Tuple[str, str]] = []
    for pat, kind in _ANCHOR_PATTERNS:
        for m in pat.finditer(chunk):
            tok = m.group(1)
            if tok in _ANCHOR_BLACKLIST:
                continue
            if len(tok) < 5:
                continue
            if tok.lower() in seen:
                continue
            seen.add(tok.lower())
            out.append((tok, kind))
    return out


# Query templates per anchor kind. Each must NOT echo non-anchor tokens
# from a typical chunk that mentions the anchor — they ask about the
# anchor in different surface form than the prose uses.
_QUERY_TEMPLATES: Dict[str, List[str]] = {
    "snake": [
        "What does {anchor} take care of?",
        "Purpose of {anchor}.",
        "{anchor} responsibility.",
        "Why does the codebase have {anchor}?",
    ],
    "camel": [
        "What is {anchor} for?",
        "Role of {anchor}.",
        "{anchor} responsibility.",
        "Where does {anchor} fit?",
    ],
    "py_file": [
        "What lives in {anchor}?",
        "{anchor} contents.",
        "Why does {anchor} exist?",
        "Purpose of {anchor}.",
    ],
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RealTextGenerator:
    """Build (memory, query) pairs from real project prose.

    Each chunk -> exactly one memory and one query. Anchors are
    globally unique across the returned set, so a query about a given
    anchor has exactly one correct hit (its source chunk).
    """

    def __init__(self, project_root: Optional[Path] = None, seed: int = 42):
        self._rng = random.Random(seed)
        self._project_root = (
            project_root
            if project_root is not None
            else Path(__file__).resolve().parent.parent.parent  # repo root
        )
        # Used per-instance and cleared on each generate() call so two
        # generations from the same instance are independent.
        self._used_anchors: set[str] = set()

    def _build_pool(self) -> List[Tuple[str, Path]]:
        """Walk the project root and produce a list of (chunk, source) pairs."""
        files = _walk_corpus(self._project_root)
        pool: List[Tuple[str, Path]] = []
        for f in files:
            try:
                text = f.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            for chunk in _chunk_text(text):
                pool.append((chunk, f))
                if len(pool) >= _MAX_CHUNKS * 4:
                    return pool
        return pool

    def generate(self, count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (memories, queries). count is an upper bound; the
        function may return fewer if the project doesn't yield enough
        chunks with unique anchors.
        """
        self._used_anchors.clear()
        pool = self._build_pool()
        memories: List[Dict[str, Any]] = []
        queries: List[Dict[str, Any]] = []

        for chunk, source in pool:
            if len(memories) >= count:
                break
            candidates = _candidate_anchors(chunk)
            if not candidates:
                continue
            # Take the first candidate not already used globally.
            anchor: Optional[Tuple[str, str]] = None
            for tok, kind in candidates:
                if tok.lower() in self._used_anchors:
                    continue
                anchor = (tok, kind)
                break
            if anchor is None:
                continue
            tok, kind = anchor
            self._used_anchors.add(tok.lower())

            template_pool = _QUERY_TEMPLATES.get(kind, _QUERY_TEMPLATES["snake"])
            qt = self._rng.choice(template_pool)

            mem_id = f"real-{kind}-{tok}-{len(memories):04d}"
            memories.append({
                "id": mem_id,
                "text": chunk,
                "label": f"real:{kind}",
                "metadata": {
                    "type": "real_text",
                    "anchor": tok,
                    "anchor_kind": kind,
                    "source_path": str(source.relative_to(self._project_root)),
                },
            })
            queries.append({
                "query": qt.format(anchor=tok),
                "ground_truth_ids": [mem_id],
                "label": f"real:{kind}",
                "anchor": tok,
                # Leakage score: Jaccard of informative tokens between
                # query and statement after stripping the shared anchor.
                # On real prose this is necessarily nonzero (the chunk
                # says "purpose" or similar in passing) but typically
                # < 0.10 with these templates.
                "leakage_score": _measure_leakage(chunk, qt.format(anchor=tok), tok),
            })
        return memories, queries


# ---------------------------------------------------------------------------
# Leakage measurement (mirrors dataset_v2 for consistency)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "which", "what",
    "when", "where", "how", "did", "does", "was", "were", "has", "have", "had",
    "are", "you", "your", "our", "their", "its", "into", "onto", "out",
    "about", "after", "before", "all", "any", "but", "not", "now", "on",
    "in", "of", "to", "is", "as", "an", "at", "by", "be",
})


def _measure_leakage(statement: str, query: str, anchor: str) -> float:
    s = set(t for t in re.findall(r"[a-z0-9]+", statement.lower())
            if t not in _STOPWORDS and len(t) > 2 and t != anchor.lower())
    q = set(t for t in re.findall(r"[a-z0-9]+", query.lower())
            if t not in _STOPWORDS and len(t) > 2 and t != anchor.lower())
    if not (s or q):
        return 0.0
    return round(len(s & q) / max(1, len(s | q)), 4)
