#!/bin/bash
# codex_orchestrator_daemon.sh — always-on continuity-keeper.
#
# Per Tito 2026-05-02: ALWAYS ON AUTONOMOUS. Codex gpt-5.5 (1M context)
# becomes the parallel orchestrator that owns continuity-keeping for the
# buildout. Aggregates reviewers, watchers, builder commits, bridge state.
# Reports to claude-code-${LANE} (the builder) via bridge + state file.
#
# Architecture:
#   - Daemon loops every POLL_INTERVAL seconds (cheap bash, no API calls)
#   - On each tick, checks if anything material changed (new commit, new
#     reviewer file, new bridge message, watcher-health change)
#   - When change detected: dispatches codex_orchestrator_synthesize.sh
#     (which fires gpt-5.5 with full context)
#   - Rate-limited: minimum MIN_SYNTH_INTERVAL between consecutive synth
#     runs even if events keep coming, so a rapid commit burst doesn't
#     melt the API budget.
#
# Idle = no codex calls. Active period = ~1 codex call per change-cluster.
#
# Run via launchd: com.ae.codex-orchestrator.plist (KeepAlive=true).
# Stop with: launchctl bootout gui/$UID com.ae.codex-orchestrator

set -uo pipefail

LANE="${NM_LANE:-nm-builder}"
POLL_INTERVAL="${ORCH_POLL_INTERVAL:-90}"
MIN_SYNTH_INTERVAL="${ORCH_MIN_SYNTH_INTERVAL:-300}"  # 5 min minimum between codex calls

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
STATE_DIR="${HOME}/.neural_memory/codex-orchestrator/${LANE}"
SYNTH="${REPO}/tools/codex_orchestrator_synthesize.sh"
LOG="${HOME}/.neural_memory/logs/codex-orchestrator-${LANE}.log"

mkdir -p "$STATE_DIR" "$(dirname "$LOG")"

LAST_HEAD_F="${STATE_DIR}/.last-git-head"
LAST_REVIEW_TS_F="${STATE_DIR}/.last-review-ts"
LAST_BRIDGE_COUNT_F="${STATE_DIR}/.last-bridge-count"
LAST_SYNTH_TS_F="${STATE_DIR}/.last-synth-ts"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "DAEMON START lane=${LANE} poll=${POLL_INTERVAL}s min-synth=${MIN_SYNTH_INTERVAL}s"

while true; do
    NOW=$(date +%s)

    # 1. Detect change in git HEAD
    CUR_HEAD=$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)
    LAST_HEAD=$(cat "$LAST_HEAD_F" 2>/dev/null || echo "")

    # 2. Detect new reviewer files
    LAST_REVIEW_TS=$(cat "$LAST_REVIEW_TS_F" 2>/dev/null || echo 0)
    NEW_REVIEW_COUNT=$(find ~/.neural_memory/per-commit-reviews ~/.neural_memory/reconciliation-reviews \
                            -type f -newermt "@${LAST_REVIEW_TS}" 2>/dev/null | wc -l | tr -d ' ')

    # 3. Detect new bridge messages addressed to my caller agent
    BRIDGE_INBOX="${HOME}/.hermes/agent-bridge/agents/claude-code-${LANE}/inbox.jsonl"
    CUR_BRIDGE_COUNT=$(wc -l < "$BRIDGE_INBOX" 2>/dev/null | tr -d ' ' || echo 0)
    LAST_BRIDGE_COUNT=$(cat "$LAST_BRIDGE_COUNT_F" 2>/dev/null || echo 0)

    # Decide if any change is material
    CHANGED=0
    REASON=""
    if [ "$CUR_HEAD" != "$LAST_HEAD" ] && [ -n "$LAST_HEAD" ]; then
        CHANGED=1
        REASON="${REASON}HEAD ${LAST_HEAD:0:7}->${CUR_HEAD:0:7}; "
    elif [ -z "$LAST_HEAD" ]; then
        # First run — synthesize once to bootstrap
        CHANGED=1
        REASON="${REASON}first-run-bootstrap; "
    fi
    if [ "$NEW_REVIEW_COUNT" -gt 0 ]; then
        CHANGED=1
        REASON="${REASON}${NEW_REVIEW_COUNT} new reviewer file(s); "
    fi
    if [ "$CUR_BRIDGE_COUNT" -gt "$LAST_BRIDGE_COUNT" ]; then
        DELTA=$((CUR_BRIDGE_COUNT - LAST_BRIDGE_COUNT))
        CHANGED=1
        REASON="${REASON}${DELTA} new bridge msg(s); "
    fi

    if [ "$CHANGED" = "1" ]; then
        # Rate-limit: respect MIN_SYNTH_INTERVAL
        LAST_SYNTH_TS=$(cat "$LAST_SYNTH_TS_F" 2>/dev/null || echo 0)
        SINCE_LAST_SYNTH=$((NOW - LAST_SYNTH_TS))
        if [ "$SINCE_LAST_SYNTH" -lt "$MIN_SYNTH_INTERVAL" ]; then
            log "SKIP synth — change detected (${REASON}) but rate-limit (${SINCE_LAST_SYNTH}s < ${MIN_SYNTH_INTERVAL}s)"
        else
            log "FIRE synth — ${REASON}"
            NM_LANE="$LANE" "$SYNTH" >> "$LOG" 2>&1
            RC=$?
            log "synth exit=${RC}"
            echo "$NOW" > "$LAST_SYNTH_TS_F"
            # Update tracking AFTER successful synth
            echo "$CUR_HEAD" > "$LAST_HEAD_F"
            echo "$NOW" > "$LAST_REVIEW_TS_F"
            echo "$CUR_BRIDGE_COUNT" > "$LAST_BRIDGE_COUNT_F"
        fi
    fi

    sleep "$POLL_INTERVAL"
done
