"""Agent self-portrait substrate read helpers.

Per the agent self-portrait feature handoff (project_self_portrait_feature_handoff_2026-05-02.md):
each AE agent (Claude Code, Valiendo/Hermes, Codex) reads its OWN substrate every 6h,
composes its OWN diffusion prompt + written reasoning, generates a visual self-image,
and stores it. This module is STEP 1 of that cycle: agent-agnostic substrate query
helpers the cycle dispatcher calls per-agent.

All helpers are READ-ONLY against the substrate. Defensive against missing data:
- If an attribution surface is empty for an agent → empty list (not crash).
- If kind='self_portrait' doesn't exist as a memory kind yet (it doesn't, in the
  live schema as of 2026-05-03) → empty list, not error.
- If no peer portraits exist yet (they won't, before the first cycle) → empty dict.

ATTRIBUTION NOTE — IMPORTANT:
The substrate has no canonical `metadata.author` field. Attribution today is
fragmented across:
  1. `origin_system` column (e.g., 'hermes', 'claude_memory', 'ae', 'cli',
     'entity_extractor', 'dream_engine', 'neural_memory')
  2. `source` column (e.g., 'bridge_mailbox', 'valiendo_handoffs',
     'claude_memory', 'dashboard')
  3. `metadata_json.from` for bridge_mailbox rows (values: 'codex',
     'claude-code', 'valiendo-hermes', 'codex-desktop', etc.)
  4. `metadata_json.author` / `metadata_json.actor` — DO NOT EXIST today, but
     supported here so the forthcoming self_portrait writer (separate packet)
     can adopt either as canonical and these helpers will pick it up.

The matcher tolerates common aliases:
  - 'claude-code' / 'claude_code' / 'claude_memory'
  - 'valiendo' / 'hermes' / 'valiendo-hermes'
  - 'codex' / 'codex-desktop'
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribution helpers
# ---------------------------------------------------------------------------

# Aliases mapped to the canonical agent name token. These are best-effort —
# the cycle dispatcher passes whatever string Tito declared canonical; this
# table just makes the matcher tolerate the natural spellings in live data.
_AGENT_ALIASES: dict[str, frozenset[str]] = {
    "claude-code": frozenset({
        "claude-code", "claude_code", "claude_memory", "claudecode", "claude code",
    }),
    "valiendo": frozenset({
        "valiendo", "valiendo-hermes", "valiendo_hermes", "hermes",
    }),
    "codex": frozenset({
        "codex", "codex-desktop", "codex_desktop", "codex-orchestrator",
    }),
}


def _alias_set(agent_name: str) -> frozenset[str]:
    """Return the alias set for a given agent_name (case-insensitive lookup).

    If the agent_name is unknown, returns a singleton set containing the
    lowercased name itself — so callers that pass arbitrary strings still get
    a usable match (filter on exactly that token).
    """
    if not agent_name:
        return frozenset()
    key = agent_name.strip().lower()
    for canonical, aliases in _AGENT_ALIASES.items():
        if key == canonical or key in aliases:
            return aliases
    return frozenset({key})


def _row_attribution_tokens(row: dict[str, Any]) -> set[str]:
    """Extract every plausible attribution token from a memory row dict.

    Reads origin_system, source, metadata_json.from, metadata_json.author,
    metadata_json.actor, metadata_json.agent. All values lowercased; missing
    fields silently skipped. Tolerates non-dict metadata_json (returns the
    empty set for that key, not a crash).
    """
    tokens: set[str] = set()
    for col in ("origin_system", "source"):
        v = row.get(col)
        if v:
            tokens.add(str(v).strip().lower())
    raw_meta = row.get("metadata_json")
    meta: dict[str, Any] = {}
    if isinstance(raw_meta, str) and raw_meta:
        try:
            parsed = json.loads(raw_meta)
            if isinstance(parsed, dict):
                meta = parsed
        except (ValueError, TypeError):
            pass
    elif isinstance(raw_meta, dict):
        meta = raw_meta
    for key in ("from", "author", "actor", "agent", "agent_name"):
        v = meta.get(key)
        if v:
            tokens.add(str(v).strip().lower())
    return tokens


def _row_matches_agent(row: dict[str, Any], agent_aliases: frozenset[str]) -> bool:
    if not agent_aliases:
        return False
    return bool(_row_attribution_tokens(row) & agent_aliases)


# ---------------------------------------------------------------------------
# Substrate access — uses NeuralMemory's existing sqlite connection, read-only.
# We do NOT open a separate read-only URI because the live process already
# holds a writable connection (SQLiteStore opens with WAL); a second ro
# connection would just mirror the same db. We simply never issue writes here.
# ---------------------------------------------------------------------------


_BASE_COLS = (
    "id, label, content, kind, salience, created_at, last_accessed, "
    "origin_system, source, metadata_json"
)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "label": row[1],
        "content": row[2],
        "kind": row[3],
        "salience": row[4],
        "ts": row[5],  # created_at, exposed as 'ts' per packet spec
        "created_at": row[5],
        "last_accessed": row[6],
        "origin_system": row[7],
        "source": row[8],
        "metadata_json": row[9],
    }


def _safe_execute(
    mem: Any, sql: str, params: tuple = ()
) -> list[tuple]:
    """Run a read-only SELECT against mem.store.conn. Defensive: returns []
    on any sqlite error rather than letting the cycle crash.

    Why defensive: a self-portrait cycle that raises stops ALL agents from
    portraying. Better to log + return empty than to break the always-on loop.
    """
    try:
        store = getattr(mem, "store", None)
        if store is None:
            return []
        conn = getattr(store, "conn", None)
        if conn is None:
            return []
        lock = getattr(store, "_lock", None)
        if lock is not None:
            with lock:
                return conn.execute(sql, params).fetchall()
        return conn.execute(sql, params).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_portrait_substrate query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _build_agent_sql_predicate(aliases: frozenset[str]) -> tuple[str, list[Any]]:
    """Build a SQL WHERE-clause fragment + params that matches origin_system OR
    source against any alias, plus a LIKE-fallback against metadata_json for
    metadata.from / .author / .actor / .agent (best-effort; no JSON1 needed).

    Returns ("(...)", [params]) — caller embeds this into a larger WHERE.
    Empty aliases → ("0", []) (matches nothing).

    Why SQL-level filter not post-filter: claude-code in the live substrate
    has 1240 rows with origin_system='claude_memory', but they're old —
    a top-200-by-created_at window catches zero of them. SQL-level filter
    means the LIMIT applies AFTER attribution, which is what callers want.
    """
    if not aliases:
        return "0", []
    parts: list[str] = []
    params: list[Any] = []
    for alias in aliases:
        parts.append("LOWER(origin_system) = ?")
        params.append(alias)
        parts.append("LOWER(source) = ?")
        params.append(alias)
        # Cheap LIKE-on-JSON-string for metadata.from / .author / .actor /
        # .agent. Quoted match avoids cross-key collisions (e.g. LIKE '%codex%'
        # would also hit "codex-orchestrator" content; we want a key:value
        # match). Using "key": "alias" pattern with a leading quote.
        for key in ("from", "author", "actor", "agent", "agent_name"):
            parts.append("metadata_json LIKE ?")
            params.append(f'%"{key}": "{alias}"%')
    return "(" + " OR ".join(parts) + ")", params


def read_self_relevant_memories(
    mem: Any, agent_name: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Return recent memories attributable to agent_name.

    Filters via SQL-level alias predicate (see _build_agent_sql_predicate)
    against origin_system / source / metadata_json LIKE on .from/.author/
    .actor/.agent. Multi-surface — metadata.author isn't sole authority yet
    (see module docstring).

    Excludes kind='entity' (derived registry rows, not authored memories)
    and kind IN ('self_portrait', 'reflection') (those are surfaced via
    read_recent_reflections so the self-portrait input bundle doesn't
    duplicate them).
    """
    aliases = _alias_set(agent_name)
    if not aliases or limit <= 0:
        return []
    pred, pred_params = _build_agent_sql_predicate(aliases)
    rows = _safe_execute(
        mem,
        f"SELECT {_BASE_COLS} FROM memories "
        "WHERE kind NOT IN ('entity', 'self_portrait', 'reflection') "
        f"  AND {pred} "
        "ORDER BY created_at DESC LIMIT ?",
        tuple(pred_params) + (limit,),
    )
    return [_row_to_dict(row) for row in rows]


def read_recent_reflections(
    mem: Any, agent_name: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Return recent reflections / self_portrait memories authored by agent_name.

    These kinds may not exist in the live schema yet (kind='self_portrait' is
    new this feature). Tolerated: query returns rows with whatever kinds
    actually exist, post-filter strips agent attribution. Empty result is fine.
    """
    aliases = _alias_set(agent_name)
    if not aliases or limit <= 0:
        return []
    pred, pred_params = _build_agent_sql_predicate(aliases)
    rows = _safe_execute(
        mem,
        f"SELECT {_BASE_COLS} FROM memories "
        "WHERE kind IN ('reflection', 'self_portrait') "
        f"  AND {pred} "
        "ORDER BY created_at DESC LIMIT ?",
        tuple(pred_params) + (limit,),
    )
    return [_row_to_dict(row) for row in rows]


def read_top_entities(
    mem: Any, agent_name: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Return entity memories most-connected to agent_name's recent activity.

    Strategy:
      1. Pull recent agent-attributed memory ids (over-fetch 100).
      2. Sum connections.weight grouped by the OTHER endpoint where
         memories.kind='entity'.
      3. Order by sum(weight) DESC, return top N as enriched memory dicts.
    """
    aliases = _alias_set(agent_name)
    if not aliases or limit <= 0:
        return []

    # Step 1: agent's recent memory ids (SQL-level filter so we don't burn
    # the LIMIT on non-agent rows).
    pred, pred_params = _build_agent_sql_predicate(aliases)
    recent_rows = _safe_execute(
        mem,
        f"SELECT {_BASE_COLS} FROM memories "
        "WHERE kind != 'entity' "
        f"  AND {pred} "
        "ORDER BY created_at DESC LIMIT 200",
        tuple(pred_params),
    )
    recent_ids: list[int] = [int(_row_to_dict(row)["id"]) for row in recent_rows]
    if not recent_ids:
        return []

    placeholders = ",".join("?" * len(recent_ids))
    # Step 2 + 3: join connections to find entity neighbors and rank by
    # summed edge weight. We consider both directions (source/target) since
    # the graph is undirected for retrieval purposes.
    sql = (
        "SELECT m.id, m.label, m.content, m.kind, m.salience, m.created_at, "
        "       m.last_accessed, m.origin_system, m.source, m.metadata_json, "
        "       SUM(c.weight) AS edge_weight_sum "
        "FROM connections c "
        "JOIN memories m ON ( "
        "  (c.source_id = m.id AND c.target_id IN (" + placeholders + ")) "
        "  OR "
        "  (c.target_id = m.id AND c.source_id IN (" + placeholders + ")) "
        ") "
        "WHERE m.kind = 'entity' AND m.id NOT IN (" + placeholders + ") "
        "GROUP BY m.id "
        "ORDER BY edge_weight_sum DESC "
        "LIMIT ?"
    )
    params = tuple(recent_ids) + tuple(recent_ids) + tuple(recent_ids) + (limit,)
    rows = _safe_execute(mem, sql, params)
    out: list[dict[str, Any]] = []
    for row in rows:
        d = _row_to_dict(row[:10])
        d["edge_weight_sum"] = row[10]
        out.append(d)
    return out


def read_recent_dream_insights(
    mem: Any, limit: int = 5
) -> list[dict[str, Any]]:
    """Return most-recent dream insight memories (D5 phase output).

    NOT agent-specific — every agent sees every insight, since dreams
    consolidate the whole substrate.

    Note: the live D5 writer uses kind='dream_insight' (per dream_engine.py
    + memory_types.py). The packet spec says kind='insight'; we accept
    BOTH for resilience in case the canonical name is renamed.
    """
    if limit <= 0:
        return []
    rows = _safe_execute(
        mem,
        f"SELECT {_BASE_COLS} FROM memories "
        "WHERE kind IN ('dream_insight', 'insight') "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


def read_peer_portraits(
    mem: Any, exclude_agent: str, limit: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Return most-recent self_portrait memories from peer agents.

    Returns dict keyed by canonical peer agent name with up to `limit` portraits
    each. Tolerates the (current) case where no portraits exist yet → empty dict.

    `exclude_agent` (and its aliases) are stripped from the result so an agent
    looking at peers never sees itself.
    """
    if limit <= 0:
        return {}
    excluded_aliases = _alias_set(exclude_agent)

    rows = _safe_execute(
        mem,
        f"SELECT {_BASE_COLS} FROM memories "
        "WHERE kind = 'self_portrait' "
        "ORDER BY created_at DESC LIMIT ?",
        (max(limit * 10, 30),),
    )
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        d = _row_to_dict(row)
        tokens = _row_attribution_tokens(d)
        if tokens & excluded_aliases:
            continue
        # Resolve to canonical name when possible; else use the first token.
        canonical: Optional[str] = None
        for cname, aliases in _AGENT_ALIASES.items():
            if tokens & aliases:
                canonical = cname
                break
        if canonical is None:
            if not tokens:
                continue
            canonical = sorted(tokens)[0]
        bucket = by_agent.setdefault(canonical, [])
        if len(bucket) < limit:
            bucket.append(d)
    return by_agent


def compose_substrate_packet(mem: Any, agent_name: str) -> dict[str, Any]:
    """Orchestrator helper: bundle every substrate read into one structured dict.

    The cycle dispatcher (separate packet) hands this dict to the agent's
    reflection step. Shape is fixed — every key always present, even if
    empty — so the dispatcher and the agent prompts can rely on it.
    """
    return {
        "agent": agent_name,
        "ts": time.time(),
        "self_memories": read_self_relevant_memories(mem, agent_name),
        "self_reflections": read_recent_reflections(mem, agent_name),
        "top_entities": read_top_entities(mem, agent_name),
        "dream_insights": read_recent_dream_insights(mem),
        "peer_portraits": read_peer_portraits(mem, agent_name),
    }
