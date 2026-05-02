#!/bin/bash
# codex_orchestrator_synthesize.sh — fire one codex gpt-5.5 synthesis pass.
#
# Called by the always-on daemon (codex_orchestrator_daemon.sh) when an
# event is detected. Bundles current state + recent activity into a single
# context payload and asks gpt-5.5 (1.05M context window) to produce a
# unified state synthesis.
#
# This is the "brain stem" layer: aggregates what reviewers/watchers/builder
# already produced, doesn't replace them. Reads everything; writes one file.
#
# Output:
#   ~/.neural_memory/codex-orchestrator/${LANE}/<ts>-state.md
#   ~/.neural_memory/codex-orchestrator/${LANE}/current-state.md (symlink)
#
# Bridge: posts FYI to claude-code-${LANE} with file pointer + summary.

set -uo pipefail

LANE="${NM_LANE:-nm-builder}"
MODEL="${ORCHESTRATOR_MODEL:-gpt-5.5}"
REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
STATE_DIR="${HOME}/.neural_memory/codex-orchestrator/${LANE}"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"

ORCH_AGENT="codex-orchestrator-${LANE}"
CALLER_AGENT="claude-code-${LANE}"

mkdir -p "$STATE_DIR"

TS=$(date +%Y%m%d-%H%M%S)
NEW_STATE="${STATE_DIR}/${TS}-state.md"
PREV_STATE="${STATE_DIR}/current-state.md"
CONTEXT_BUNDLE=$(mktemp -t codex-orch-ctx.XXXXXX.md)

# --- Build context bundle ---
{
    echo "# Codex Orchestrator Context Bundle"
    echo "Lane: ${LANE}"
    echo "Synth time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""

    echo "## Current git HEAD + last 30 commits"
    cd "$REPO" && git log --oneline -30 2>&1
    echo ""

    echo "## Working tree status"
    cd "$REPO" && git status --short 2>&1
    echo ""

    echo "## Save-state file (anchor for what's active)"
    cat ~/.claude/projects/-Users-tito/memory/project_nm_session_2026-05-02_save_state.md 2>/dev/null
    echo ""

    echo "## Last 5 per-commit-reviewer findings"
    for f in $(ls -t ~/.neural_memory/per-commit-reviews/*.md 2>/dev/null | head -5); do
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## Last 5 reconciliation-reviewer findings"
    for f in $(ls -t ~/.neural_memory/reconciliation-reviews/*.md 2>/dev/null | head -5); do
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## Last 30 lines of watcher-health log"
    tail -30 ~/.neural_memory/logs/watcher-health.log 2>/dev/null
    echo ""

    echo "## Recent bridge digest (claude-code-${LANE})"
    if [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
        node "$BRIDGE_CLI" digest --agent "$CALLER_AGENT" --limit 15 2>/dev/null | head -200
    fi
    echo ""

    echo "## Previous orchestrator state (for diff-aware synthesis)"
    if [ -f "$PREV_STATE" ]; then
        cat "$PREV_STATE"
    else
        echo "(no previous state — this is the first synthesis pass)"
    fi
    echo ""
} > "$CONTEXT_BUNDLE"

# --- Build the orchestrator prompt ---
PROMPT="You are the codex-orchestrator for the ${LANE} lane of the Angels Electric / neural-memory buildout. Your job is continuity-keeping: synthesize the current state from the bundle below.

Read the entire context bundle. Then produce a markdown state document with these sections:

## Stable
What's working and not changing. One line per item.

## Changed since last synthesis
What's new since the previous orchestrator state file (if present). Each item: file/system + nature of change + commit/source ref.

## Open issues across components
Cross-cutting concerns: reviewer findings not yet acted on, drift between modules, things flagged but not closed. Reference the source.

## Anomalies
Unexpected state. Reviewer logs gone silent, watchers reporting stale, commits without tests, bridge messages without acks.

## Suggested next moves for the builder
Ranked list, evidence-grounded only — point to specific files/lines. NO speculation. If you don't have evidence for a recommendation, don't make it.

## Health check on the orchestrator itself
Are you (the orchestrator) seeing the inputs you should? If watcher-health log is empty or per-commit-reviews dir is missing, flag that — your continuity-keeping function depends on those inputs.

Be terse. Be evidence-grounded. Cite source files for every claim. If something is unclear or you lack evidence, write 'insufficient evidence' rather than guess.

Context bundle follows in the file at: ${CONTEXT_BUNDLE}

Read that file with: cat ${CONTEXT_BUNDLE}"

# --- Run codex with read-only sandbox ---
{
    echo "# Codex Orchestrator State — ${LANE}"
    echo "**Synth time:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Model:** ${MODEL}"
    echo "**Repo HEAD:** $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)"
    echo "**Context bundle:** ${CONTEXT_BUNDLE}"
    echo ""
    echo "---"
    echo ""
} > "$NEW_STATE"

"$CODEX" exec \
    --model "$MODEL" \
    --sandbox read-only \
    --cd "$REPO" \
    "$PROMPT" 2>&1 | tee -a "$NEW_STATE"
RC=${PIPESTATUS[0]}

# Update current-state symlink
ln -sf "$NEW_STATE" "${STATE_DIR}/current-state.md.tmp" && mv "${STATE_DIR}/current-state.md.tmp" "$PREV_STATE"

# Bridge FYI
if [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
    SUMMARY=$(grep -E '^##|^- ' "$NEW_STATE" | head -8 | head -c 500)
    [ -z "$SUMMARY" ] && SUMMARY="(no structured output)"
    BODY="Codex orchestrator synthesis complete.
Lane: ${LANE}
Model: ${MODEL}
Exit: ${RC}
File: ${NEW_STATE}
Quick view: ${PREV_STATE} (symlink to latest)

Top sections / items:
${SUMMARY}"
    node "$BRIDGE_CLI" send \
        --from "$ORCH_AGENT" \
        --to "$CALLER_AGENT" \
        --subject "orchestrator/${LANE}: state synth (HEAD $(cd "$REPO" && git rev-parse --short HEAD), exit=${RC})" \
        --body "$BODY" \
        --urgency low \
        2>/dev/null || true
fi

# Cleanup tmp bundle (state file keeps the relevant content)
rm -f "$CONTEXT_BUNDLE"

echo ""
echo "→ saved: $NEW_STATE"
echo "→ symlink: $PREV_STATE"
exit $RC
