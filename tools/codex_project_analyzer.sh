#!/bin/bash
# codex_project_analyzer.sh — single-shot gpt-5.5 deep-read of the project,
# producing a compact primer doc that orchestrator + reconciler include in
# their bundles to ground every synthesis in project context.
#
# Per Tito 2026-05-02: "MAKE SURE THAT THE GPT 5.5S ARE UP TO SPEED ON THE
# PROJECT PLEASE. OR CREATE A GPT 5.5. TO ANALYZE THE ENTIRE THING + REPORT
# TO ORCHESTRATOR + RECONCILER."
#
# Architecture:
#   - This subagent runs RARELY (weekly cron + on-demand)
#   - Reads heavy: full memory dir + repo structure + git log + active state
#   - Writes ONE compact output: project-primer.md (~20K-40K tokens curated)
#   - Orchestrator + reconciler include the primer in their bundles every run
#     (cheap, since primer is small and stable)
#
# This avoids bundle-bloat-on-every-synth while still keeping all gpt-5.5
# instances project-aware.
#
# Output: ~/.neural_memory/codex-orchestrator/project-primer.md
# (intentionally NOT lane-namespaced — primer is shared across consumers)
#
# Usage:
#   tools/codex_project_analyzer.sh
#
# Cron: weekly via com.ae.codex-project-analyzer.plist

set -uo pipefail

LANE="${NM_LANE:-shared}"
MODEL="${ANALYZER_MODEL:-gpt-5.5}"
REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
OUT_DIR="${HOME}/.neural_memory/codex-orchestrator"
PRIMER="${OUT_DIR}/project-primer.md"
PREV_PRIMER="${OUT_DIR}/project-primer.previous.md"
LOG="${HOME}/.neural_memory/logs/codex-project-analyzer.log"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"
BUNDLE=$(mktemp -t codex-analyzer-bundle.XXXXXX.md)

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "ANALYZER START lane=${LANE} model=${MODEL}"

# Backup previous primer
[ -f "$PRIMER" ] && cp "$PRIMER" "$PREV_PRIMER"

# --- Build the deep-read bundle ---
{
    echo "# Project Deep-Read Bundle"
    echo "Lane: ${LANE}"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""

    echo "## Master rules (CLAUDE.md)"
    cat /Users/tito/.claude/CLAUDE.md 2>/dev/null
    echo ""

    echo "## Memory index (MEMORY.md)"
    cat ~/.claude/projects/-Users-tito/memory/MEMORY.md 2>/dev/null
    echo ""

    echo "## All project_*.md memory files (full content)"
    for f in ~/.claude/projects/-Users-tito/memory/project_*.md; do
        [ -f "$f" ] || continue
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## All feedback_*.md memory files (full content) — META RULES"
    for f in ~/.claude/projects/-Users-tito/memory/feedback_*.md; do
        [ -f "$f" ] || continue
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## All reference_*.md memory files (full content)"
    for f in ~/.claude/projects/-Users-tito/memory/reference_*.md; do
        [ -f "$f" ] || continue
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## All skill_*.md memory files (full content)"
    for f in ~/.claude/projects/-Users-tito/memory/skill_*.md; do
        [ -f "$f" ] || continue
        echo "### $(basename "$f")"
        cat "$f"
        echo ""
    done

    echo "## Repo structure (top 3 levels)"
    cd "$REPO" && find . -maxdepth 3 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.plist' \) -not -path './.*' -not -path './node_modules/*' -not -path './.venv/*' 2>/dev/null | head -200

    echo ""
    echo "## Full commit history (last 200, oneline)"
    cd "$REPO" && git log --oneline -200 2>/dev/null

    echo ""
    echo "## Repo README (if present)"
    if [ -f "$REPO/README.md" ]; then
        head -300 "$REPO/README.md"
    else
        echo "(none)"
    fi

    echo ""
    echo "## Previous primer (for diff awareness)"
    if [ -f "$PREV_PRIMER" ]; then
        cat "$PREV_PRIMER"
    else
        echo "(no previous primer — first analyzer run)"
    fi
} > "$BUNDLE"

BUNDLE_TOKENS=$(wc -c < "$BUNDLE")
log "Bundle built: $(du -h "$BUNDLE" | cut -f1) ($BUNDLE_TOKENS chars)"

PROMPT="You are the codex-project-analyzer subagent. Your job: produce a CONCISE project primer that brings any gpt-5.5 instance (orchestrator, reconciler, or future codex callers) up to speed on this project.

You have a deep-read bundle in: ${BUNDLE}

Use 'cat ${BUNDLE}' to read it. Then produce a markdown primer with EXACTLY these sections, in this order:

# Project Primer — neural-memory + Angels Electric ecosystem

## What this project is (1 paragraph)
The single sentence answer to 'what are we building'. Then 2-3 sentences of context.

## Why it exists (1 paragraph)
The motivation. Who uses it, what problem it solves, what success looks like.

## Architecture (top-level)
- Components and their roles (5-8 bullets)
- Data flow / control flow (1 short ASCII diagram if helpful)
- What's intentionally NOT here (anti-scope)

## Active topology (right now)
- Builder layer: who/what
- Orchestrator layer: who/what
- Subagent layer: per-commit reviewer + reconciliation reviewer + archaeology cron + watcher-health + neural-observer (with current models)
- Always-on daemons: list with cadence
- Bridge agents registered: list

## Current state of the work
- What's stable
- What's in flight (recent commits, open WIP)
- What's deferred

## Key decisions / DO NOT list
- Architectural choices that should NOT be re-litigated (with rationale)
- DO NOTs that protect against past mistakes

## Reading order for new gpt-5.5 instances
- 'If you only read 5 files, read these 5'
- File paths + 1-line reason for each

## Open questions / future work
- Bullet list of what's queued

KEEP IT TIGHT. Target: 8,000-15,000 words MAX. The whole point is to be a STABLE compact reference. Avoid:
- Listing every commit (use 'last 20' style)
- Reciting memory files verbatim (synthesize across them)
- Speculation (you have evidence — use [verified-now] tags for cited claims, [inferred] for synthesis)

You are READ-ONLY (sandbox enforces). Output your primer as markdown to stdout — wrapper captures it to project-primer.md."

# --- Header ---
{
    echo "# Project Primer — neural-memory + Angels Electric ecosystem"
    echo ""
    echo "**Generated:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Generator:** codex-project-analyzer (${MODEL})"
    echo "**Repo HEAD:** $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)"
    echo "**Bundle size:** $(du -h "$BUNDLE" | cut -f1)"
    echo "**Refresh cadence:** weekly via com.ae.codex-project-analyzer.plist (or on-demand: tools/codex_project_analyzer.sh)"
    echo "**Consumers:** orchestrator daemon (every synth), reconciliation reviewer (every prompt)"
    echo ""
    echo "---"
    echo ""
} > "$PRIMER.tmp"

log "Calling codex ${MODEL}..."

"$CODEX" exec \
    --model "$MODEL" \
    --sandbox read-only \
    --cd "$REPO" \
    "$PROMPT" \
    >> "$PRIMER.tmp" 2>> "$LOG"
RC=$?

if [ $RC -eq 0 ] && [ -s "$PRIMER.tmp" ]; then
    mv "$PRIMER.tmp" "$PRIMER"
    log "ANALYZER OK exit=$RC primer=$PRIMER ($(wc -c < "$PRIMER") chars)"
else
    log "ANALYZER FAIL exit=$RC — primer NOT updated, prev primer kept at $PREV_PRIMER"
    rm -f "$PRIMER.tmp"
fi

# Bridge FYI
if [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
    node "$BRIDGE_CLI" send \
        --from "codex-project-analyzer" \
        --to "claude-code-${LANE}" \
        --subject "project-analyzer: primer refreshed (lane=${LANE}, exit=${RC})" \
        --body "Primer file: ${PRIMER}
Bundle size: $(du -h "$BUNDLE" 2>/dev/null | cut -f1)
Output size: $(du -h "$PRIMER" 2>/dev/null | cut -f1 || echo 'N/A')
Consumers (orchestrator + reconciler) will pick up the new primer on their next bundle build." \
        --urgency low \
        2>/dev/null || true
fi

rm -f "$BUNDLE"
exit $RC
