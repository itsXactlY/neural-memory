#!/bin/bash
# per_commit_reviewer.sh — run a codex gpt-5.4 reviewer over each new commit
# on the active NM-lane branch since the last reviewer tick.
#
# Migrated 2026-05-02 from claude-sonnet-4-6 to codex gpt-5.4 per Tito directive
# ("EVERYTHING SHOULD HAVE BEEN MIGRATED to this latest codex/hermes agent setup
# with gpt 5.5 codex orchestrator always on in parallel w buildout. EVERYTHING
# unless required to be anthropic model subagent"). Per-commit review is single-
# commit scope so gpt-5.4 is sufficient; gpt-5.5 reserved for reconciliation.
#
# What it does:
#   1. Reads ~/.neural_memory/per-commit-reviewer-state (stored last-reviewed-SHA)
#   2. git log --reverse <last-sha>..HEAD on the active branch
#   3. For each new commit, dispatches codex gpt-5.4 reviewer with a focused
#      review prompt; captures stdout as findings file
#   4. Updates last-reviewed-SHA to HEAD
#
# Usage (manual):
#   tools/per_commit_reviewer.sh
#
# Cron:
#   See tools/launchd/com.ae.neural-memory-per-commit-review.plist (5min)

set -euo pipefail

REPO_DIR="/Users/tito/lWORKSPACEl/research/neural-memory"
STATE_FILE="${HOME}/.neural_memory/per-commit-reviewer-state"
FINDINGS_DIR="${HOME}/.neural_memory/per-commit-reviews"
LOG_FILE="${HOME}/.neural_memory/logs/per-commit-reviewer.log"
CODEX_BIN="/Applications/Codex.app/Contents/Resources/codex"
MODEL="${PER_COMMIT_MODEL:-gpt-5.4}"

mkdir -p "$(dirname "$STATE_FILE")" "$FINDINGS_DIR" "$(dirname "$LOG_FILE")"

cd "$REPO_DIR"

CURRENT_HEAD=$(git rev-parse HEAD)
LAST_SHA=""
if [ -f "$STATE_FILE" ]; then
    LAST_SHA=$(cat "$STATE_FILE")
fi

if [ -z "$LAST_SHA" ]; then
    # First run: only review HEAD (don't replay all of history)
    NEW_COMMITS="$CURRENT_HEAD"
    echo "[$(date)] First run; reviewing HEAD ($CURRENT_HEAD) only" >> "$LOG_FILE"
elif [ "$LAST_SHA" = "$CURRENT_HEAD" ]; then
    echo "[$(date)] No new commits since $LAST_SHA — skipping" >> "$LOG_FILE"
    exit 0
else
    NEW_COMMITS=$(git log --reverse --pretty=format:%H "${LAST_SHA}..HEAD")
fi

REVIEWED=0
for SHA in $NEW_COMMITS; do
    SHORT=$(git rev-parse --short "$SHA")
    SUBJECT=$(git log -1 --pretty=format:%s "$SHA")
    OUT="${FINDINGS_DIR}/${SHORT}-review.md"

    if [ -f "$OUT" ]; then
        echo "[$(date)] $SHORT already reviewed at $OUT — skipping" >> "$LOG_FILE"
        continue
    fi

    PROMPT="You are a code-correctness reviewer for the neural-memory project at ${REPO_DIR}.

Review commit ${SHA} (subject: '${SUBJECT}'). Use 'git show ${SHA}' to see the diff.

Look for:
1. Real bugs that would cause crashes or wrong behavior in production
2. Tests added (if any) — do they actually exercise the wired code path?
3. Schema/storage changes — do they violate the Phase 7 'don't touch substrate' boundary?
4. Performance concerns — new SQL, new loops, unbounded growth
5. Inconsistency with existing Phase 7.5 wiring patterns (α/β/γ/δ/ε)

Output your report as markdown to stdout. Begin with '# Review: ${SHORT} — ${SUBJECT}' as h1. Under 200 words. [verified-now] tag for confirmed issues. file_path:line cites. Be terse and evidence-grounded. Don't manufacture findings — if commit is clean, say PASS in one line."

    echo "[$(date)] Dispatching codex ${MODEL} review for $SHORT ($SUBJECT)" >> "$LOG_FILE"
    {
        echo "# Review: $SHORT — $SUBJECT"
        echo "**Reviewer:** codex ${MODEL}"
        echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo ""
    } > "$OUT"
    "$CODEX_BIN" exec \
        --model "$MODEL" \
        --sandbox read-only \
        --cd "$REPO_DIR" \
        "$PROMPT" \
        >> "$OUT" 2>> "$LOG_FILE" \
        || echo "[$(date)] WARN: review for $SHORT exited non-zero" >> "$LOG_FILE"

    if [ -s "$OUT" ]; then
        echo "[$(date)] Review for $SHORT landed at $OUT" >> "$LOG_FILE"
        REVIEWED=$((REVIEWED + 1))
    fi
done

echo "$CURRENT_HEAD" > "$STATE_FILE"
echo "[$(date)] Reviewed $REVIEWED commits; state updated to $CURRENT_HEAD" >> "$LOG_FILE"
