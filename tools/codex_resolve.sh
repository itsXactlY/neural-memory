#!/bin/bash
# codex_resolve.sh — given an escalation, dispatch codex gpt-5.5 to propose
# a fix patch. Closes the orchestrator → builder loop with a NEXT step
# (proposed patch the builder can review/apply) instead of just a finding.
#
# Per Tito 2026-05-02: "compound its usefulness/buildout synergy" —
# don't just escalate problems, propose fixes for them.
#
# Pattern:
#   - Orchestrator escalates N issues to claude-code-${LANE} via high-urgency bridge
#   - Wrapper detects esc>0 + dispatches this script per escalation item
#   - codex_resolve reads escalation text + relevant code (cited file:line)
#   - Produces structured patch proposal in markdown (diff blocks)
#   - Posts bridge FYI from codex-resolver-${LANE} → claude-code-${LANE}
#
# Builder still owns approval/apply — this is a proposer, not an applier.
# Codex sandbox is read-only; can't mutate code. Proposal lives at
# ~/.neural_memory/codex-resolutions/${LANE}/<ts>-<topic>.md
#
# Usage:
#   tools/codex_resolve.sh <topic> "<escalation-text>"
#   echo "<escalation>" | tools/codex_resolve.sh <topic>

set -uo pipefail

TOPIC="${1:-untitled}"
ESC_TEXT="${2:-}"
LANE="${NM_LANE:-nm-builder}"
MODEL="${RESOLVER_MODEL:-gpt-5.5}"

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
OUT_DIR="${HOME}/.neural_memory/codex-resolutions/${LANE}"
PRIMER="${HOME}/.neural_memory/codex-orchestrator/project-primer.md"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"

RESOLVER_AGENT="codex-resolver-${LANE}"
CALLER_AGENT="claude-code-${LANE}"

mkdir -p "$OUT_DIR"

if [ -z "$ESC_TEXT" ]; then
    if [ -t 0 ]; then
        echo "ERROR: no escalation text and no stdin pipe" >&2
        exit 2
    fi
    ESC_TEXT="$(cat)"
fi

TOPIC_SAFE=$(echo "$TOPIC" | tr -c 'a-zA-Z0-9_-' '_' | head -c 40)
TS=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_DIR}/${TS}-${TOPIC_SAFE}.md"

# Bracket the prompt with primer reference + escalation + structured-output ask
PRIMER_PREAMBLE=""
if [ -f "$PRIMER" ]; then
    PRIMER_PREAMBLE="Read the project primer at ${PRIMER} (use 'cat ${PRIMER}') for project context.

"
fi

PROMPT="${PRIMER_PREAMBLE}You are the codex-resolver subagent for the ${LANE} lane. The orchestrator just escalated a finding to the builder. Your job: read the cited code, understand the issue, propose a concrete fix as a unified diff that the builder can review + apply.

ESCALATION TEXT:
${ESC_TEXT}

INSTRUCTIONS:
1. For each cited file:line, read the surrounding code (use 'sed -n', 'cat', or 'rg').
2. Verify the issue is real before proposing a fix. If you can't reproduce or the citation is stale, say 'INSUFFICIENT EVIDENCE — cannot verify' and explain.
3. If real, propose a minimal unified-diff patch. Format:
   \`\`\`diff
   --- a/path/to/file
   +++ b/path/to/file
   @@ ... @@
   - old line
   + new line
   \`\`\`
4. Explain WHY the fix works in 1-2 sentences (what root cause it addresses).
5. Note any RISKS (other code paths affected, tests that may need updating).
6. End with a one-line VERDICT: 'READY-TO-APPLY' (high confidence, minimal blast radius) or 'NEEDS-REVIEW' (ambiguous, complex, or wide blast radius).

CONSTRAINTS:
- Sandbox is read-only — you cannot mutate code. Propose the patch; builder applies.
- Keep diffs MINIMAL — change only what's needed. No drive-by refactors.
- If escalation is informational (e.g., 'Tito needs to restart Hermes' — not a code fix), say 'NOT-A-CODE-FIX — informational escalation, no patch needed'.
- Be evidence-grounded. Cite the file:line you read for every claim.

Output format: markdown with the diff block + explanation + risks + verdict."

# Header
{
    echo "# Codex Resolution Proposal — ${TOPIC}"
    echo "**Lane:** ${LANE}"
    echo "**Model:** ${MODEL}"
    echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Repo HEAD:** $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)"
    echo ""
    echo "## Escalation that triggered this"
    echo ""
    echo "${ESC_TEXT}" | head -50
    echo ""
    echo "## Proposed resolution"
    echo ""
} > "${OUT}.partial"

"$CODEX" exec \
    --model "$MODEL" \
    --sandbox read-only \
    --cd "$REPO" \
    "$PROMPT" \
    >> "${OUT}.partial" 2>/dev/null
RC=$?

# Only finalize if codex produced meaningful output
if [ "$RC" = "0" ] && [ "$(wc -c < "${OUT}.partial" 2>/dev/null || echo 0)" -gt 800 ]; then
    mv "${OUT}.partial" "$OUT"
else
    # Keep partial for diagnosis
    echo ""
    echo "**Exit:** $RC (resolver may have failed — see partial file: ${OUT}.partial)" >> "${OUT}.partial"
    OUT="${OUT}.partial"
fi

# Extract verdict for bridge subject
VERDICT=$(grep -oE "READY-TO-APPLY|NEEDS-REVIEW|NOT-A-CODE-FIX|INSUFFICIENT EVIDENCE" "$OUT" 2>/dev/null | tail -1)
[ -z "$VERDICT" ] && VERDICT="UNKNOWN"

# Bridge FYI
if [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
    BODY="Codex resolver produced a fix proposal for escalation: ${TOPIC}
Lane: ${LANE}
Model: ${MODEL}
Verdict: ${VERDICT}
File: ${OUT}

The orchestrator escalated; resolver proposed a patch. Builder review + apply if VERDICT=READY-TO-APPLY. Use 'cat ${OUT}' to see the diff."
    URGENCY="low"
    [ "$VERDICT" = "READY-TO-APPLY" ] && URGENCY="normal"
    node "$BRIDGE_CLI" send \
        --from "$RESOLVER_AGENT" \
        --to "$CALLER_AGENT" \
        --subject "resolver/${LANE}: ${VERDICT} — ${TOPIC}" \
        --body "$BODY" \
        --urgency "$URGENCY" \
        2>/dev/null || true
fi

echo ""
echo "→ saved: $OUT"
echo "→ verdict: $VERDICT"
exit $RC
