#!/usr/bin/env python3
"""self_portrait_cycle.py — orchestrator for STEPS 2-7 of the agent self-portrait
cycle (per S-PORTRAIT-DESIGN spec, 2026-05-03).

Per the design spec PART B: each AE agent (claude-code, valiendo/hermes, codex)
reads its OWN substrate every 6 hours, composes its OWN diffusion prompt + written
reasoning, generates a visual self-image, and stores everything to the substrate.

This module is the orchestrator — STEPS 2 through 7. STEP 1 (substrate read) lives
in python/self_portrait_substrate.py and is invoked here. STEP 5 (image-gen) calls
the OpenAI Images API directly (no MCP bridge). STEP 6 (diff-from-prior) is a
deterministic text comparison for v0; the design spec's Sonnet-subagent diff is
deferred to P1.

Tito hard rules (non-negotiable):
  1. The orchestrator MUST NOT pre-fill, template, or constrain the agent's
     diffusion prompt. The agent decides everything; we just pass `prompt_text`
     through verbatim.
  2. Reasoning is mandatory each cycle alongside the visual.
  3. References are inspiration only, not templates.
  4. Always-on, no manual triggers. (Cron is a separate packet.)

Two invocation modes:

  --mode scaffold  (default)
      Runs STEP 2 (substrate read) and writes the substrate packet to the agent's
      input.json under ~/.neural_memory/portraits/<agent>/cycle-<ts>/input.json.
      No image generation. No store. The agent picks up input.json on its next
      turn, composes its own reasoning + prompt, and re-invokes us in --mode
      complete with those values.

  --mode complete
      Requires --reasoning-text and --prompt-text (both agent-authored). Runs
      STEPS 4-7: validates the prompt, generates the image, computes the diff
      from the prior cycle, and stores the portrait row.

CLI:
    python3 tools/self_portrait_cycle.py --agent <agent_name> [--mode scaffold|complete]
                                          [--reasoning-text <text>] [--prompt-text <text>]
                                          [--db <path>] [--dry-run]

Notes:
  - Only stdlib used for image-gen (urllib.request + json) per packet constraint.
  - On missing OPENAI_API_KEY or API failure: log warning, set image_path=None,
    cycle still completes with reasoning + prompt stored. The cycle never crashes
    on image-gen failure (always-on requirement).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Allow importing python/* siblings whether invoked as module or script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Image-gen defaults. The handoff doc named "gpt-image-2 via Hermes openai-codex
# plugin" — that's a Hermes-side alias. Since this orchestrator calls OpenAI
# directly (bridge OFF per NM topology), defaults use the real OpenAI model
# name + a supported size. Per Tito hard rule #1, agents pick their own model
# preferences anyway — these are just the defaults if an agent doesn't override
# via --image-model / --image-size.
#   model: gpt-image-1 (released Apr 2025, current OpenAI image model)
#   size:  1536x1024 (landscape, closest supported to aLca's 1920x1088)
_DEFAULT_IMAGE_MODEL = "gpt-image-1"
_DEFAULT_IMAGE_SIZE = "1536x1024"
_OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"
_OPENAI_TIMEOUT_S = 30.0
_OPENAI_DOWNLOAD_TIMEOUT_S = 30.0

# Agent-authored prompt validation per packet acceptance criteria.
_MAX_PROMPT_LEN = 4000

# Where per-agent portrait artifacts land. Cycle-scoped subdirs hold the
# substrate packet (input.json) so the agent can re-read what it reflected on.
_PORTRAITS_ROOT = Path.home() / ".neural_memory" / "portraits"

# Default substrate path (mirrors memory_client.DB_PATH default).
_DEFAULT_DB_PATH = Path.home() / ".neural_memory" / "memory.db"

# Salience for self-portrait rows (per packet: medium-high, intentional).
_PORTRAIT_SALIENCE = 0.8
_PORTRAIT_CONFIDENCE = 1.0


# ---------------------------------------------------------------------------
# STEP 2: Substrate read scaffold
# ---------------------------------------------------------------------------


def write_substrate_input_packet(
    packet: dict[str, Any], agent_name: str, cycle_ts: float
) -> Path:
    """Persist the substrate packet so the agent can ingest it next turn.

    Returns the absolute path of the written input.json. Cycle-scoped subdir
    (cycle-<int_ts>) keeps successive cycles separate without overwriting.
    """
    cycle_dir = _PORTRAITS_ROOT / agent_name / f"cycle-{int(cycle_ts)}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    input_path = cycle_dir / "input.json"
    input_path.write_text(json.dumps(packet, indent=2, default=str))
    return input_path


def run_scaffold_mode(
    mem: Any, agent_name: str, cycle_ts: float
) -> dict[str, Any]:
    """STEP 2 only: read substrate, write input packet, return summary.

    The agent picks up input.json on its next turn, composes reasoning + prompt,
    and re-invokes the cycle in --mode complete.
    """
    from self_portrait_substrate import compose_substrate_packet  # noqa: WPS433

    packet = compose_substrate_packet(mem, agent_name)
    input_path = write_substrate_input_packet(packet, agent_name, cycle_ts)
    return {
        "mode": "scaffold",
        "agent_name": agent_name,
        "cycle_ts": cycle_ts,
        "substrate_packet_path": str(input_path),
        "packet_keys": sorted(packet.keys()),
        "self_memories_count": len(packet.get("self_memories", [])),
        "peer_portraits_agents": sorted(packet.get("peer_portraits", {}).keys()),
    }


# ---------------------------------------------------------------------------
# STEP 4: Validate agent-authored prompt (no templating, no pre-fill)
# ---------------------------------------------------------------------------


def validate_agent_prompt(prompt_text: str) -> str:
    """Validate the agent-authored prompt. Returns the prompt unchanged on success.

    Tito hard rule #1: orchestrator does NOT modify, template, or pre-fill the
    prompt. Validation is bounds-only: non-empty after strip, length <= 4000.
    """
    if not isinstance(prompt_text, str):
        raise ValueError("prompt_text must be a string")
    stripped = prompt_text.strip()
    if not stripped:
        raise ValueError("prompt_text must be non-empty (agent must author it)")
    if len(prompt_text) > _MAX_PROMPT_LEN:
        raise ValueError(
            f"prompt_text exceeds max length {_MAX_PROMPT_LEN} "
            f"(got {len(prompt_text)})"
        )
    return prompt_text


def validate_agent_reasoning(reasoning_text: str) -> str:
    """Validate agent-authored reasoning. Bounds-only (Tito rule #2: mandatory)."""
    if not isinstance(reasoning_text, str):
        raise ValueError("reasoning_text must be a string")
    stripped = reasoning_text.strip()
    if not stripped:
        raise ValueError(
            "reasoning_text must be non-empty (Tito rule #2: reasoning mandatory)"
        )
    return reasoning_text


# ---------------------------------------------------------------------------
# STEP 5: Image generation — direct HTTPS to OpenAI Images API
# ---------------------------------------------------------------------------


class _ImageGenResult:
    """Tiny container so we can capture path + url + model in one return."""

    __slots__ = ("image_path", "image_url", "model_used", "error")

    def __init__(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        model_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.image_path = image_path
        self.image_url = image_url
        self.model_used = model_used
        self.error = error


def _post_openai_images(
    prompt_text: str,
    api_key: str,
    model: str,
    size: str,
    *,
    url: str = _OPENAI_IMAGES_URL,
    timeout: float = _OPENAI_TIMEOUT_S,
) -> dict[str, Any]:
    """POST to OpenAI Images API. Returns parsed JSON. Raises on HTTP/network error."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_text,
            "n": 1,
            "size": size,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _download_image(url: str, dest: Path, timeout: float = _OPENAI_DOWNLOAD_TIMEOUT_S) -> None:
    """Download image bytes from a URL to dest. Raises on failure."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)


def generate_image(
    prompt_text: str,
    agent_name: str,
    cycle_ts: float,
    *,
    model: str = _DEFAULT_IMAGE_MODEL,
    size: str = _DEFAULT_IMAGE_SIZE,
    portraits_root: Path = _PORTRAITS_ROOT,
) -> _ImageGenResult:
    """Generate the agent's self-portrait image.

    Graceful-failure contract (always-on requirement): on missing API key,
    network error, HTTP 4xx/5xx, or any unexpected exception, log a warning
    and return _ImageGenResult with image_path=None. The cycle proceeds.

    On success: PNG saved to portraits_root/<agent_name>/<int_ts>.png (or
    .jpg/.bin if no clear extension can be derived; OpenAI typically returns
    PNG).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set; skipping image generation for %s cycle %s",
            agent_name,
            cycle_ts,
        )
        return _ImageGenResult(model_used=model, error="OPENAI_API_KEY not set")

    portrait_dir = portraits_root / agent_name
    portrait_dir.mkdir(parents=True, exist_ok=True)
    image_path = portrait_dir / f"{int(cycle_ts)}.png"

    try:
        data = _post_openai_images(prompt_text, api_key, model, size)
    except urllib.error.HTTPError as exc:
        logger.warning(
            "OpenAI Images API HTTP %s for %s: %s", exc.code, agent_name, exc.reason
        )
        return _ImageGenResult(model_used=model, error=f"HTTP {exc.code}")
    except urllib.error.URLError as exc:
        logger.warning("OpenAI Images API network error for %s: %s", agent_name, exc)
        return _ImageGenResult(model_used=model, error=f"URLError: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OpenAI Images API unexpected failure for %s: %s", agent_name, exc
        )
        return _ImageGenResult(model_used=model, error=f"unexpected: {exc}")

    # OpenAI Images API returns either {"data": [{"url": ...}]} or
    # {"data": [{"b64_json": ...}]} depending on response_format / model. We
    # try url-first, then b64. If neither, treat as failure.
    try:
        item = data["data"][0]
    except (KeyError, IndexError, TypeError):
        logger.warning(
            "OpenAI Images API returned unexpected shape for %s: %r", agent_name, data
        )
        return _ImageGenResult(model_used=model, error="unexpected response shape")

    image_url = item.get("url")
    b64 = item.get("b64_json")

    if image_url:
        try:
            _download_image(image_url, image_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Image download failed for %s from %s: %s",
                agent_name,
                image_url,
                exc,
            )
            # We have a URL but couldn't download — still return URL for the
            # store row so the agent can retry/fetch later.
            return _ImageGenResult(
                image_path=None,
                image_url=image_url,
                model_used=model,
                error=f"download failed: {exc}",
            )
        return _ImageGenResult(
            image_path=str(image_path), image_url=image_url, model_used=model
        )

    if b64:
        import base64

        try:
            image_path.write_bytes(base64.b64decode(b64))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Base64 image decode failed for %s: %s", agent_name, exc
            )
            return _ImageGenResult(
                model_used=model, error=f"b64 decode failed: {exc}"
            )
        return _ImageGenResult(
            image_path=str(image_path), image_url=None, model_used=model
        )

    logger.warning(
        "OpenAI Images API returned no url and no b64_json for %s", agent_name
    )
    return _ImageGenResult(model_used=model, error="empty response")


# ---------------------------------------------------------------------------
# STEP 6: Diff-from-prior (deterministic; Sonnet-subagent diff deferred to P1)
# ---------------------------------------------------------------------------


def _read_prior_reasoning(mem: Any, agent_name: str) -> Optional[str]:
    """Find the most-recent kind='self_portrait' row authored by agent_name and
    return its reasoning_text. Reasoning lives on the row's `content` column
    per STEP 7 store contract; metadata.reasoning_text is the historical
    fallback.

    Returns None if no prior portrait exists or substrate is unreachable.
    """
    store = getattr(mem, "store", None)
    conn = getattr(store, "conn", None) if store is not None else None
    if conn is None:
        return None
    try:
        # Query mirrors the substrate-read alias matching shape: origin_system
        # OR metadata_json.author / .agent_name pointing at agent_name. Cheap
        # LIKE is acceptable here because we only need MAX(created_at) for one
        # agent at a time.
        rows = conn.execute(
            """
            SELECT content, metadata_json
            FROM memories
            WHERE kind = 'self_portrait'
              AND (
                LOWER(origin_system) = ?
                OR metadata_json LIKE ?
                OR metadata_json LIKE ?
              )
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                agent_name.lower(),
                f'%"author": "{agent_name}"%',
                f'%"agent_name": "{agent_name}"%',
            ),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("prior portrait lookup failed for %s: %s", agent_name, exc)
        return None
    if not rows:
        return None
    content, metadata_json = rows[0]
    if content:
        return content
    if metadata_json:
        try:
            md = json.loads(metadata_json)
            return md.get("reasoning_text")
        except Exception:  # noqa: BLE001
            return None
    return None


def _diff_summary(prior: Optional[str], current: str) -> str:
    """Deterministic diff summary (v0).

    Per packet: "NOT Sonnet calling itself — it's a deterministic text
    comparison". Full Sonnet-subagent diff is a P1 follow-up.

    Strategy: token-set Jaccard similarity over alphanumeric word tokens.
    - No prior  → "First portrait cycle for {agent_name}." (caller wraps)
    - Identical → "Reasoning stable; same themes."
    - High overlap (>= 0.7) → "Mostly stable; small additions/removals."
    - Moderate (0.3 .. 0.7) → "Notable shift; added: <new>; dropped: <gone>."
    - Low (< 0.3)           → "Major shift; <n_kept> shared tokens."

    Token lists are truncated to first 5 to keep the summary one sentence.
    """
    if not current:
        return "Reasoning empty; no diff produced."
    if prior is None:
        # The caller (run_complete_mode) supplies the agent name in its
        # own first-portrait branch. This generic fallback is here so
        # _diff_summary is also safe when called outside that wrapper.
        return "First portrait cycle (no prior reasoning found)."

    def _tokens(text: str) -> set[str]:
        return {tok.lower() for tok in _split_words(text) if len(tok) > 2}

    def _split_words(text: str) -> list[str]:
        # Stdlib only; alphanumeric split via str.isalnum walk.
        out: list[str] = []
        buf: list[str] = []
        for ch in text:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    out.append("".join(buf))
                    buf.clear()
        if buf:
            out.append("".join(buf))
        return out

    prior_tok = _tokens(prior)
    cur_tok = _tokens(current)
    if not prior_tok and not cur_tok:
        return "Reasoning stable; no comparable tokens."
    union = prior_tok | cur_tok
    inter = prior_tok & cur_tok
    jaccard = len(inter) / len(union) if union else 1.0

    if prior.strip() == current.strip():
        return "Reasoning stable; same themes."
    if jaccard >= 0.7:
        return "Mostly stable; small additions/removals."
    added = sorted(cur_tok - prior_tok)[:5]
    dropped = sorted(prior_tok - cur_tok)[:5]
    if 0.3 <= jaccard < 0.7:
        return (
            f"Notable shift; added: {', '.join(added) or 'none'}; "
            f"dropped: {', '.join(dropped) or 'none'}."
        )
    return f"Major shift; {len(inter)} shared tokens of {len(union)} total."


# ---------------------------------------------------------------------------
# STEP 7: Store
# ---------------------------------------------------------------------------


def _resolve_store_helper(mem: Any) -> Any:
    """Prefer NeuralMemory.remember() (auto-embeds); fall back to store.store()
    (caller-supplied embedding). Returns a callable accepting our kwargs.

    Why prefer remember(): it computes the embedding via mem.embedder so we
    don't have to plumb embed_provider through this orchestrator. Conflict-
    detection is harmless for self-portrait kind (no contradictions to mine).
    """
    if hasattr(mem, "remember"):
        return mem.remember
    store = getattr(mem, "store", None)
    if store is not None and hasattr(store, "store"):
        # store.store() requires embedding; this branch needs caller to embed
        # first — we do that in store_portrait below.
        return store.store
    raise RuntimeError("memory provider has neither remember() nor store.store()")


def store_portrait(
    mem: Any,
    *,
    agent_name: str,
    cycle_ts: float,
    reasoning_text: str,
    prompt_text: str,
    image_path: Optional[str],
    image_url: Optional[str],
    model_used: str,
    diff_from_prior_summary: str,
    substrate_packet_path: Optional[str],
    anchor_seed: Optional[int] = None,
) -> int:
    """Write a kind='self_portrait' row to the substrate per STEP 7 contract.

    Returns the new memory id (or -1 if the underlying store helper does not
    return one).
    """
    metadata = {
        "author": agent_name,  # S-PORTRAIT-1 attribution recommendation
        "agent_name": agent_name,
        "cycle_ts": cycle_ts,
        "image_path": image_path,
        "image_url": image_url,
        "prompt_text": prompt_text,  # agent-authored, stored verbatim
        "model_used": model_used or _DEFAULT_IMAGE_MODEL,
        "anchor_seed": anchor_seed,
        "diff_from_prior": diff_from_prior_summary,
        "substrate_packet_path": substrate_packet_path,
    }

    label = f"Self-portrait: {agent_name} @ {int(cycle_ts)}"
    # content = reasoning_text per packet ("the agent's written self-reflection
    # — that's the searchable substrate content").
    content = reasoning_text

    # Prefer NeuralMemory.remember() so we don't have to call embed_provider
    # ourselves. detect_conflicts=False because self-portrait reasoning isn't a
    # factual claim to be contradiction-checked.
    if hasattr(mem, "remember"):
        new_id = mem.remember(
            content,
            label=label,
            detect_conflicts=False,
            kind="self_portrait",
            confidence=_PORTRAIT_CONFIDENCE,
            source="self_portrait_cycle",
            origin_system=agent_name,
            valid_from=cycle_ts,
            metadata=metadata,
            salience=_PORTRAIT_SALIENCE,
        )
        return int(new_id) if new_id is not None else -1

    # Fallback: SQLiteStore.store() with caller-supplied embedding. We compute
    # via embed_provider in this branch; if even that fails, raise.
    store = getattr(mem, "store")
    from embed_provider import EmbeddingProvider  # noqa: WPS433

    embedder = EmbeddingProvider(backend="auto")
    embedding = embedder.embed(content)
    new_id = store.store(
        label=label,
        content=content,
        embedding=embedding,
        kind="self_portrait",
        confidence=_PORTRAIT_CONFIDENCE,
        source="self_portrait_cycle",
        origin_system=agent_name,
        valid_from=cycle_ts,
        transaction_time=cycle_ts,
        metadata=metadata,
        salience=_PORTRAIT_SALIENCE,
    )
    return int(new_id) if new_id is not None else -1


# ---------------------------------------------------------------------------
# Mode runner: complete (STEPS 4-7 after a previous scaffold run)
# ---------------------------------------------------------------------------


def run_complete_mode(
    mem: Any,
    *,
    agent_name: str,
    reasoning_text: str,
    prompt_text: str,
    cycle_ts: float,
    dry_run: bool = False,
    image_model: str = _DEFAULT_IMAGE_MODEL,
    image_size: str = _DEFAULT_IMAGE_SIZE,
    portraits_root: Path = _PORTRAITS_ROOT,
    substrate_packet_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run STEPS 4-7. Caller has already produced reasoning + prompt (--mode
    complete). Returns a summary dict (also useful for tests).
    """
    # STEP 4: validate (no templating, no pre-fill).
    validate_agent_reasoning(reasoning_text)
    validate_agent_prompt(prompt_text)

    # STEP 5: generate image (graceful failure → image_path/url=None).
    image_result = generate_image(
        prompt_text,
        agent_name,
        cycle_ts,
        model=image_model,
        size=image_size,
        portraits_root=portraits_root,
    )

    # STEP 6: diff-from-prior. First-portrait branch uses agent name.
    prior_reasoning = _read_prior_reasoning(mem, agent_name)
    if prior_reasoning is None:
        diff_summary = f"First portrait cycle for {agent_name}."
    else:
        diff_summary = _diff_summary(prior_reasoning, reasoning_text)

    # STEP 7: store (skipped under --dry-run).
    if dry_run:
        new_id = -1
        stored = False
    else:
        new_id = store_portrait(
            mem,
            agent_name=agent_name,
            cycle_ts=cycle_ts,
            reasoning_text=reasoning_text,
            prompt_text=prompt_text,
            image_path=image_result.image_path,
            image_url=image_result.image_url,
            model_used=image_result.model_used or image_model,
            diff_from_prior_summary=diff_summary,
            substrate_packet_path=substrate_packet_path,
        )
        stored = True

    return {
        "mode": "complete",
        "agent_name": agent_name,
        "cycle_ts": cycle_ts,
        "image_path": image_result.image_path,
        "image_url": image_result.image_url,
        "image_error": image_result.error,
        "model_used": image_result.model_used or image_model,
        "diff_from_prior": diff_summary,
        "memory_id": new_id,
        "stored": stored,
        "dry_run": dry_run,
        "substrate_packet_path": substrate_packet_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent self-portrait cycle orchestrator (STEPS 2-7). "
            "Default mode 'scaffold' writes the substrate input packet only; "
            "'complete' requires --reasoning-text + --prompt-text and runs "
            "STEPS 4-7 (image-gen + store)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name (canonical: 'claude-code' | 'valiendo' | 'codex').",
    )
    parser.add_argument(
        "--mode",
        choices=("scaffold", "complete"),
        default="scaffold",
        help="scaffold = write substrate input only; complete = run STEPS 4-7.",
    )
    parser.add_argument(
        "--reasoning-text",
        default=None,
        help=(
            "Agent-authored reasoning text (Tito rule #2: mandatory). "
            "Required in --mode complete."
        ),
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help=(
            "Agent-authored diffusion prompt (Tito rule #1: agent picks own "
            "aesthetic — orchestrator does NOT pre-fill or template). "
            "Required in --mode complete."
        ),
    )
    parser.add_argument(
        "--db",
        default=str(_DEFAULT_DB_PATH),
        help="Substrate path (overrides default ~/.neural_memory/memory.db).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the substrate write at STEP 7 (still does image-gen).",
    )
    parser.add_argument(
        "--image-model",
        default=_DEFAULT_IMAGE_MODEL,
        help="Image model identifier passed to OpenAI Images API.",
    )
    parser.add_argument(
        "--image-size",
        default=_DEFAULT_IMAGE_SIZE,
        help="Image size passed to OpenAI Images API.",
    )
    parser.add_argument(
        "--portraits-root",
        default=str(_PORTRAITS_ROOT),
        help="Root directory for per-agent portrait artifacts.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def _open_memory(db_path: str) -> Any:
    """Open NeuralMemory bound to the given db_path. Lazy-imported because
    NeuralMemory pulls in heavy deps (sentence-transformers, hnswlib, etc.)
    that we don't want to cost --help startup.
    """
    from memory_client import NeuralMemory  # noqa: WPS433

    return NeuralMemory(db_path=db_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "complete":
        if not args.reasoning_text or not args.prompt_text:
            parser.error(
                "--mode complete requires both --reasoning-text and --prompt-text "
                "(Tito rule #2: reasoning mandatory; rule #1: agent-authored prompt)."
            )

    portraits_root = Path(args.portraits_root)
    cycle_ts = time.time()

    mem = _open_memory(args.db)

    if args.mode == "scaffold":
        # Override the module-level _PORTRAITS_ROOT for the scaffold writer
        # by passing through a wrapper: write_substrate_input_packet uses the
        # module constant, so we temporarily monkey it. Cleaner: derive cycle
        # dir inline.
        from self_portrait_substrate import compose_substrate_packet  # noqa: WPS433

        packet = compose_substrate_packet(mem, args.agent)
        cycle_dir = portraits_root / args.agent / f"cycle-{int(cycle_ts)}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        input_path = cycle_dir / "input.json"
        input_path.write_text(json.dumps(packet, indent=2, default=str))
        result = {
            "mode": "scaffold",
            "agent_name": args.agent,
            "cycle_ts": cycle_ts,
            "substrate_packet_path": str(input_path),
            "packet_keys": sorted(packet.keys()),
            "self_memories_count": len(packet.get("self_memories", [])),
            "peer_portraits_agents": sorted(packet.get("peer_portraits", {}).keys()),
        }
    else:
        # Best-effort: locate a recent scaffold input.json for this agent.
        substrate_packet_path: Optional[str] = None
        agent_root = portraits_root / args.agent
        if agent_root.exists():
            cycles = sorted(
                (p for p in agent_root.iterdir() if p.is_dir() and p.name.startswith("cycle-")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if cycles:
                candidate = cycles[0] / "input.json"
                if candidate.exists():
                    substrate_packet_path = str(candidate)

        result = run_complete_mode(
            mem,
            agent_name=args.agent,
            reasoning_text=args.reasoning_text,
            prompt_text=args.prompt_text,
            cycle_ts=cycle_ts,
            dry_run=args.dry_run,
            image_model=args.image_model,
            image_size=args.image_size,
            portraits_root=portraits_root,
            substrate_packet_path=substrate_packet_path,
        )

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
