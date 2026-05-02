#!/bin/bash
# codex_subagent.sh — route auxiliary subagent work to Codex (GPT) instead
# of spawning Claude subagents. Per Tito 2026-05-02:
#
#   "only truly want always on reviewers/reconcilers that need to be claude
#    code agent subagents + builder to remain this window. yet still
#    maintaining overarching continuity + collaboration via mcp bridge."
#
# Lane labeling: each builder (NM-builder, AE-builder, hermes-mirror, cron)
# runs the SAME wrapper but sets NM_LANE so their codex agent identity,
# output dir, and bridge message sender are all distinct — no collisions
# between independent claude-code instances repurposing codex in parallel.
#
# Use this for:
#   - Code archaeology / audit walks
#   - File scans ("show me everything that touches X")
#   - Research (read-N-files-and-summarize)
#   - Anything where the answer is mechanical, not interpretive
#
# Stay on Claude Code subagent for:
#   - Per-commit review (interpretive: did this commit make sense)
#   - Reconciliation review (cross-references current state vs intent)
#   - Builder synthesis (this window)
#
# Continuity: every run writes findings to disk, posts FYI to bridge.
#
# Usage:
#   NM_LANE=nm-builder tools/codex_subagent.sh archaeology memory_client "Review python/memory_client.py for stale or dead code paths"
#   NM_LANE=cron      tools/codex_subagent.sh archaeology dream_engine "..."
#   echo "prompt..." | NM_LANE=ae-builder tools/codex_subagent.sh audit fts5_sync
#
# Args: <type> <topic> [<prompt-or-stdin>]
# Env:
#   NM_LANE       — lane label (default "manual"). Used for agent name + output path.
#   CODEX_MODEL   — model name (default "gpt-5.4")
#   NM_BRIDGE_FYI — set to "0" to skip bridge FYI post (default: post)

set -uo pipefail

TYPE="${1:-archaeology}"
TOPIC="${2:-untitled}"
PROMPT="${3:-}"
LANE="${NM_LANE:-manual}"
BRIDGE_FYI="${NM_BRIDGE_FYI:-1}"

# Type → default model selection. Override via CODEX_MODEL env var.
# Available codex models (per CLI, 2026-05-02): gpt-5.5, gpt-5.4, gpt-5.4-mini,
# gpt-5.3-codex, plus 3 others. Both gpt-5.5 and gpt-5.4 have identical
# 1,050,000-token context windows + 128,000-token max output (per OpenAI docs
# 2026-05-02, confirmed by tito) — so model choice is about REASONING DEPTH
# and cost-per-call, NOT prompt-size fit. Even whole-module + callers fits
# comfortably in either.
#   gpt-5.5         — deeper reasoning, more expensive (cross-cutting audits)
#   gpt-5.4         — general balanced default
#   gpt-5.4-mini    — fast cheap (simple lookups, scans)
#   gpt-5.3-codex   — code-specialized (refactor drafts, code-shape critique)
case "$TYPE" in
    archaeology)  DEFAULT_MODEL="gpt-5.4" ;;     # docstring/behavior drift on one file
    audit)        DEFAULT_MODEL="gpt-5.5" ;;     # cross-cutting concerns benefit from biggest model
    scan)         DEFAULT_MODEL="gpt-5.4-mini" ;; # mechanical inventory — fast/cheap
    research)     DEFAULT_MODEL="gpt-5.4" ;;     # read-N-files-and-summarize
    code)         DEFAULT_MODEL="gpt-5.3-codex" ;; # code drafts / refactor proposals
    *)            DEFAULT_MODEL="gpt-5.4" ;;
esac
MODEL="${CODEX_MODEL:-$DEFAULT_MODEL}"

# Driver: codex (default) or hermes (not yet wired — hermes-mirror dispatch
# from bash is awkward; use bridge orchestration_request for stateful/tool-using
# tasks that need Hermes).
DRIVER="${NM_DRIVER:-codex}"
if [ "$DRIVER" != "codex" ]; then
    echo "ERROR: NM_DRIVER=$DRIVER not implemented in this wrapper." >&2
    echo "       For Hermes/Valiendo subagent work, use bridge orchestration_request" >&2
    echo "       (mcp__claude-hermes-bridge__request_orchestration_approval) — Hermes is" >&2
    echo "       better suited to stateful, multi-turn, MCP-tool-using tasks." >&2
    echo "       This wrapper is for stateless mechanical work that codex handles cleanly." >&2
    exit 3
fi

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
BASEDIR="${HOME}/.neural_memory/codex-subagent-runs/${LANE}"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"

CODEX_AGENT="codex-via-${LANE}"
CALLER_AGENT="claude-code-${LANE}"

mkdir -p "$BASEDIR"

if [ -z "$PROMPT" ]; then
    if [ -t 0 ]; then
        echo "ERROR: no prompt argument and no stdin pipe" >&2
        echo "Usage: $0 <type> <topic> <prompt>  OR  echo 'prompt' | $0 <type> <topic>" >&2
        exit 2
    fi
    PROMPT="$(cat)"
fi

# Sanitize topic for filename (alnum + dash/underscore only, max 40 chars)
TOPIC_SAFE=$(echo "$TOPIC" | tr -c 'a-zA-Z0-9_-' '_' | head -c 40)
TS=$(date +%Y%m%d-%H%M%S)
RUN_ID="${LANE}-${TS}-$$"
OUT="${BASEDIR}/${TS}-${TYPE}-${TOPIC_SAFE}.md"

# Header
{
    echo "# Codex subagent run — ${TYPE} / ${TOPIC}"
    echo "**Lane:** ${LANE}"
    echo "**Run ID:** ${RUN_ID}"
    echo "**Model:** ${MODEL}"
    echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Repo HEAD:** $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo ""
    echo "## Prompt"
    echo ""
    echo "${PROMPT}"
    echo ""
    echo "## Codex output"
    echo ""
} > "$OUT"

# Run codex exec, read-only sandbox so it can't mutate the tree
"$CODEX" exec \
    --model "$MODEL" \
    --sandbox read-only \
    --cd "$REPO" \
    "$PROMPT" 2>&1 | tee -a "$OUT"
RC=${PIPESTATUS[0]}

{
    echo ""
    echo "---"
    echo "**Exit:** $RC"
} >> "$OUT"

# Bridge FYI — non-fatal if it fails (script doesn't depend on bridge being up)
if [ "$BRIDGE_FYI" = "1" ] && [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
    # First-line-of-output summary; fall back to "(no output)"
    SUMMARY=$(grep -v '^$' "$OUT" | grep -v '^#' | grep -v '^\*\*' | head -1 | head -c 200)
    [ -z "$SUMMARY" ] && SUMMARY="(no output)"
    BODY="Codex subagent run finished.
Lane: ${LANE}
Type: ${TYPE} / ${TOPIC}
Model: ${MODEL}
Exit: ${RC}
File: ${OUT}
First line: ${SUMMARY}"
    node "$BRIDGE_CLI" send \
        --from "$CODEX_AGENT" \
        --to "$CALLER_AGENT" \
        --subject "codex/${TYPE}: ${TOPIC} (lane=${LANE}, exit=${RC})" \
        --body "$BODY" \
        --urgency low \
        2>/dev/null || echo "  (bridge FYI post failed — non-fatal)" >&2
fi

echo ""
echo "→ saved: $OUT"
echo "→ lane: $LANE  agent: $CODEX_AGENT  caller: $CALLER_AGENT"
exit $RC
