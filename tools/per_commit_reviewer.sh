#!/bin/bash
# per_commit_reviewer.sh — run a sonnet reviewer over each new commit on the
# active NM-lane branch since the last reviewer tick.
#
# Adopted 2026-05-01 from AE-builder peer's archaeology-tick pattern:
# polling daemon that catches commits without operator intervention. AE-builder
# runs theirs at 15 min cadence; NM lane substrate changes slower so this is
# 60 min by default.
#
# What it does:
#   1. Reads ~/.neural_memory/per-commit-reviewer-state (stored last-reviewed-SHA)
#   2. git log --reverse <last-sha>..HEAD on the active branch
#   3. For each new commit, dispatches a sonnet reviewer via `claude -p` with
#      a focused review prompt; saves output to a per-commit findings file
#   4. Updates last-reviewed-SHA to HEAD
#
# Usage (manual):
#   tools/per_commit_reviewer.sh
#
# Cron (recommended):
#   See tools/launchd/com.ae.neural-memory-per-commit-review.plist

set -euo pipefail

REPO_DIR="/Users/tito/lWORKSPACEl/research/neural-memory"
STATE_FILE="${HOME}/.neural_memory/per-commit-reviewer-state"
FINDINGS_DIR="${HOME}/.neural_memory/per-commit-reviews"
LOG_FILE="${HOME}/.neural_memory/logs/per-commit-reviewer.log"

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

Report under 200 words. [verified-now] tag for confirmed issues. file_path:line cites. Save your full report to ${OUT}."

    echo "[$(date)] Dispatching sonnet review for $SHORT ($SUBJECT)" >> "$LOG_FILE"
    /Users/tito/.local/bin/claude -p \
        --model claude-sonnet-4-6 \
        "$PROMPT" \
        > "${OUT}.transcript" 2>> "$LOG_FILE" \
        || echo "[$(date)] WARN: review for $SHORT exited non-zero" >> "$LOG_FILE"

    if [ -f "$OUT" ]; then
        echo "[$(date)] Review for $SHORT landed at $OUT" >> "$LOG_FILE"
        REVIEWED=$((REVIEWED + 1))
    fi
done

echo "$CURRENT_HEAD" > "$STATE_FILE"
echo "[$(date)] Reviewed $REVIEWED commits; state updated to $CURRENT_HEAD" >> "$LOG_FILE"
