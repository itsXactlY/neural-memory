#!/bin/bash
# codex_always_on_runner.sh — wrap any periodic codex script as ALWAYS-ON.
#
# Per Tito 2026-05-02: "NOT NO TIME BASED THINGS. THE ENTIRE THING SHOULD
# BE ALWAYS ON. ALWAYS ON TIL COMPLETION IN THIS 20H+ SESSION WE ARE STILL
# GOING THROUGH."
#
# Eliminates the cron pattern entirely. Each underlying script is wrapped
# in a polling loop with: cheap change detection (free bash) + min-interval
# rate-limit (protects API budget) + auto-restart via KeepAlive=true plist.
#
# Pattern is: check for change → if changed AND interval-elapsed → fire script.
#
# Usage:
#   NM_LANE=cron AOR_NAME=archaeology AOR_SCRIPT=tools/codex_archaeology_pick.sh \
#     AOR_MIN_INTERVAL=600 AOR_POLL=120 \
#     tools/codex_always_on_runner.sh
#
# Change detection: any of git HEAD changed, target dir mtime newer than
# last-fire timestamp, or N seconds elapsed (max-interval guarantee).

set -uo pipefail

NAME="${AOR_NAME:?AOR_NAME required}"
SCRIPT="${AOR_SCRIPT:?AOR_SCRIPT required (path relative to repo root)}"
LANE="${NM_LANE:-cron}"
POLL="${AOR_POLL:-120}"               # poll cadence (cheap bash, free)
MIN_INTERVAL="${AOR_MIN_INTERVAL:-1800}"  # min seconds between codex fires (rate-limit)
MAX_INTERVAL="${AOR_MAX_INTERVAL:-7200}"  # max wait between fires even with no change (default 2h)

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
STATE_DIR="${HOME}/.neural_memory/codex-aor/${NAME}"
LOG="${HOME}/.neural_memory/logs/aor-${NAME}.log"

mkdir -p "$STATE_DIR" "$(dirname "$LOG")"

LAST_FIRE_F="${STATE_DIR}/.last-fire"
LAST_HEAD_F="${STATE_DIR}/.last-head"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [aor:${NAME}] $*" | tee -a "$LOG"; }

log "AOR START name=${NAME} script=${SCRIPT} poll=${POLL}s min-interval=${MIN_INTERVAL}s max-interval=${MAX_INTERVAL}s"

while true; do
    NOW=$(date +%s)
    LAST_FIRE=$(cat "$LAST_FIRE_F" 2>/dev/null || echo 0)
    SINCE_FIRE=$((NOW - LAST_FIRE))

    # Change detection: git HEAD diff
    CUR_HEAD=$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unknown)
    LAST_HEAD=$(cat "$LAST_HEAD_F" 2>/dev/null || echo "")
    HEAD_CHANGED=0
    if [ "$CUR_HEAD" != "$LAST_HEAD" ] && [ -n "$LAST_HEAD" ]; then
        HEAD_CHANGED=1
    fi
    FIRST_RUN=0
    [ -z "$LAST_HEAD" ] && FIRST_RUN=1

    SHOULD_FIRE=0
    REASON=""
    # Per resolver patch 2026-05-02: distinguish true first-run (LAST_FIRE=0)
    # from retry-after-failed-first-run (FIRST_RUN still true because failure
    # didn't update LAST_HEAD, but LAST_FIRE was written to gate retries).
    # Without this distinction, failed first-runs refire every $POLL bypassing
    # $MIN_INTERVAL.
    if [ "$FIRST_RUN" = "1" ] && [ "$LAST_FIRE" = "0" ]; then
        SHOULD_FIRE=1; REASON="first-run-bootstrap"
    elif [ "$FIRST_RUN" = "1" ] && [ "$SINCE_FIRE" -ge "$MIN_INTERVAL" ]; then
        SHOULD_FIRE=1; REASON="first-run-retry (${SINCE_FIRE}s ≥ ${MIN_INTERVAL}s)"
    elif [ "$SINCE_FIRE" -ge "$MAX_INTERVAL" ]; then
        SHOULD_FIRE=1; REASON="max-interval (${SINCE_FIRE}s ≥ ${MAX_INTERVAL}s)"
    elif [ "$HEAD_CHANGED" = "1" ] && [ "$SINCE_FIRE" -ge "$MIN_INTERVAL" ]; then
        SHOULD_FIRE=1; REASON="HEAD ${LAST_HEAD:0:7}->${CUR_HEAD:0:7}"
    fi

    if [ "$SHOULD_FIRE" = "1" ]; then
        log "FIRE — ${REASON}"
        NM_LANE="$LANE" "${REPO}/${SCRIPT}" >> "$LOG" 2>&1
        RC=$?
        log "exit=${RC}"
        if [ "$RC" = "0" ]; then
            echo "$NOW" > "$LAST_FIRE_F"
            echo "$CUR_HEAD" > "$LAST_HEAD_F"
        else
            # Write LAST_FIRE_F=$NOW even on failure so MIN_INTERVAL gates retries.
            # Bug caught by Sonnet hostile-reviewer 2026-05-02 — without this, persistent
            # failures (codex auth issue, transient API error) refire every $POLL tick
            # bypassing $MIN_INTERVAL and burning API budget continuously.
            # LAST_HEAD_F NOT updated — preserves the change signal for next tick to
            # retry on the same content (so we still attempt the failed work).
            echo "$NOW" > "$LAST_FIRE_F"
            log "FAILED — change signal preserved (LAST_HEAD unchanged), next retry gated by MIN_INTERVAL"
        fi
    fi

    sleep "$POLL"
done
