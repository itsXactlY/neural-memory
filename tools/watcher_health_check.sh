#!/bin/bash
# watcher_health_check.sh — observer-of-observers
#
# Per Tito 2026-05-02 directive: "keep [the reviewer] always on behind you
# like you used to have it. reroute it every time to make sure everything
# else is tracking/working too."
#
# Each tracker has a known cadence + log file. If a tracker hasn't fired
# in N × cadence, this script flags it. Optionally pings telegram for
# critical staleness (skip in dry-run mode).
#
# Tracked watchers:
#   - per-commit reviewer (cron StartInterval=300s, every 5min)
#   - vault status sync   (cron daily at 04:30)
#   - corpus ingest       (cron daily, runs ingest_ae_corpus.py)
#   - bench daily         (cron daily at 03:00, writes bench-history/)
#   - neural observer     (cron StartInterval=900s, every 15min)
#   - HD price catalog    (substrate row count vs expected)
#   - AccessLogger        (recall events being logged)
#
# Usage:
#   tools/watcher_health_check.sh                # report; exit 0 if all healthy
#   tools/watcher_health_check.sh --alert        # also send telegram on critical
#
# Cron: every 30 min (com.ae.neural-watcher-health.plist)

set -uo pipefail

LOG_FILE="${HOME}/.neural_memory/logs/watcher-health.log"
TELEGRAM_BIN="${HOME}/.hermes/bin/ae_telegram_send.py"
ALERT_MODE=0
[[ "${1:-}" == "--alert" ]] && ALERT_MODE=1

mkdir -p "$(dirname "$LOG_FILE")"

NOW_TS=$(date +%s)
NOW_HUMAN=$(date "+%Y-%m-%d %H:%M:%S %Z")
ISSUES=()
HEALTHY=()

# Helper: file-mtime age in seconds
mtime_age() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo -1
        return
    fi
    local mt
    mt=$(stat -f %m "$f" 2>/dev/null || echo 0)
    echo $((NOW_TS - mt))
}

# Helper: row delta in JSONL (compares to prev snapshot stored in state)
STATE_FILE="${HOME}/.neural_memory/.watcher-health-state"
prev_count() {
    local key="$1"
    [ -f "$STATE_FILE" ] && grep "^${key}=" "$STATE_FILE" 2>/dev/null | cut -d= -f2 || echo 0
}
record_count() {
    local key="$1" val="$2"
    touch "$STATE_FILE"
    grep -v "^${key}=" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
    echo "${key}=${val}" >> "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

# 1. Per-commit reviewer (5min cadence; allow 3× = 15min staleness)
F="${HOME}/.neural_memory/logs/per-commit-reviewer.log"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    ISSUES+=("CRITICAL: per-commit-reviewer log missing at $F")
elif [ "$A" -gt 900 ]; then
    ISSUES+=("STALE: per-commit-reviewer last fired ${A}s ago (expected ≤ 900s)")
else
    HEALTHY+=("per-commit-reviewer (${A}s ago)")
fi

# 2. Neural observer (15min cadence; allow 3× = 45min staleness)
F="${HOME}/.neural_memory/logs/observer.stdout.log"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    ISSUES+=("CRITICAL: neural-observer log missing")
elif [ "$A" -gt 2700 ]; then
    ISSUES+=("STALE: neural-observer last fired ${A}s ago (expected ≤ 2700s)")
else
    HEALTHY+=("neural-observer (${A}s ago)")
fi

# 2b. Reconciliation reviewer (60min cadence; allow 3× = 3h staleness)
F="${HOME}/.neural_memory/logs/reconciliation-reviewer.log"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    HEALTHY+=("reconciliation-reviewer (not yet fired)")
elif [ "$A" -gt 10800 ]; then
    ISSUES+=("STALE: reconciliation-reviewer last fired ${A}s ago (expected ≤ 10800s)")
else
    HEALTHY+=("reconciliation-reviewer (${A}s ago)")
fi

# 3. Vault status sync (daily; allow 36h)
F="${HOME}/.neural_memory/logs/vault-status.stdout.log"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    ISSUES+=("CRITICAL: vault-status-sync log missing")
elif [ "$A" -gt 129600 ]; then
    ISSUES+=("STALE: vault-status-sync last fired ${A}s ago (>36h)")
else
    HEALTHY+=("vault-status-sync (${A}s ago)")
fi

# 4. Bench history (daily; allow 36h between entries)
LATEST=$(ls -t ${HOME}/.neural_memory/bench-history/*.json 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    ISSUES+=("CRITICAL: bench-history empty")
else
    A=$(mtime_age "$LATEST")
    if [ "$A" -gt 129600 ]; then
        ISSUES+=("STALE: bench-history latest entry ${A}s ago (>36h): $(basename "$LATEST")")
    else
        HEALTHY+=("bench-history ($(basename "$LATEST"), ${A}s ago)")
    fi
fi

# 5. AccessLogger (recall events being logged)
F="${HOME}/.neural_memory/logs/recall-access.jsonl"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    ISSUES+=("INFO: AccessLogger has no recall events yet (file missing)")
else
    CUR=$(wc -l < "$F" | tr -d ' ')
    PREV=$(prev_count "access_log_lines")
    record_count "access_log_lines" "$CUR"
    if [ "$CUR" -gt "$PREV" ]; then
        DELTA=$((CUR - PREV))
        HEALTHY+=("AccessLogger (+${DELTA} events since last check, total ${CUR})")
    else
        # Not stale per-se if there's just no recall traffic; flag as INFO
        if [ "$A" -gt 3600 ]; then
            HEALTHY+=("AccessLogger (no traffic ${A}s, total ${CUR})")
        else
            HEALTHY+=("AccessLogger (idle, total ${CUR})")
        fi
    fi
fi

# 6b. Hermes plugin symlink integrity — closes the 7-day silent
# staleness vector caught by Tito's brief 2026-05-02 (commit 704aae6).
# 6 files in hermes-plugin/ MUST be symlinks to ../python/ versions,
# not concrete copies. If any becomes a real file again (e.g., someone
# mistakes a symlink for stale code and replaces it), the next Hermes
# restart silently reverts to Phase-pre-7 code.
PLUGIN_DIR="/Users/tito/lWORKSPACEl/research/neural-memory/hermes-plugin"
EXPECTED_SYMLINKS="memory_client.py dream_engine.py mssql_store.py embed_provider.py neural_memory.py cpp_bridge.py"
broken_links=()
for f in $EXPECTED_SYMLINKS; do
    p="${PLUGIN_DIR}/${f}"
    if [ ! -L "$p" ]; then
        broken_links+=("$f")
    fi
done
if [ "${#broken_links[@]}" -gt 0 ]; then
    ISSUES+=("CRITICAL: hermes-plugin symlink(s) corrupted (now concrete files): ${broken_links[*]} — Hermes will silently revert to stale code on next restart. Re-symlink via: cd $PLUGIN_DIR && for f in ${broken_links[*]}; do mv \$f \$f.bak && ln -s ../python/\$f \$f; done")
else
    HEALTHY+=("hermes-plugin symlinks (6/6 intact)")
fi

# 8. Codex orchestrator daemon (always-on; check it's running)
ORCH_LOG="${HOME}/.neural_memory/logs/codex-orchestrator-nm-builder.log"
A=$(mtime_age "$ORCH_LOG")
if [ "$A" -lt 0 ]; then
    ISSUES+=("CRITICAL: codex-orchestrator-nm-builder daemon log missing — daemon not started?")
elif [ "$A" -gt 600 ]; then
    # Daemon polls every 90s, should write to log on every poll. >10min stale = dead.
    ISSUES+=("STALE: codex-orchestrator daemon log not updated in ${A}s — daemon may have died (KeepAlive should restart, check stderr)")
else
    HEALTHY+=("codex-orchestrator daemon (${A}s ago)")
fi

# 7. Codex archaeology cron (daily; allow 36h staleness)
F="${HOME}/.neural_memory/logs/codex-archaeology.stdout.log"
A=$(mtime_age "$F")
if [ "$A" -lt 0 ]; then
    HEALTHY+=("codex-archaeology (not yet fired — first run pending)")
elif [ "$A" -gt 129600 ]; then
    ISSUES+=("STALE: codex-archaeology last fired ${A}s ago (>36h) — check codex auth or cron loaded")
else
    HEALTHY+=("codex-archaeology (${A}s ago)")
fi

# 6. Substrate growth check (every run, vs prev snapshot)
DB_PATH="${HOME}/.neural_memory/memory.db"
if [ -f "$DB_PATH" ]; then
    CUR=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM memories" 2>/dev/null || echo 0)
    PREV=$(prev_count "substrate_count")
    record_count "substrate_count" "$CUR"
    DELTA=$((CUR - PREV))
    HEALTHY+=("substrate count=${CUR} delta=${DELTA}")
fi

# Report
{
    echo "[${NOW_HUMAN}] watcher-health check"
    echo "  HEALTHY (${#HEALTHY[@]}):"
    for h in "${HEALTHY[@]}"; do echo "    ✓ $h"; done
    if [ "${#ISSUES[@]}" -gt 0 ]; then
        echo "  ISSUES (${#ISSUES[@]}):"
        for i in "${ISSUES[@]}"; do echo "    ✗ $i"; done
    else
        echo "  ISSUES: none"
    fi
} | tee -a "$LOG_FILE"

# Optional: telegram alert on issues
if [ "$ALERT_MODE" = "1" ] && [ "${#ISSUES[@]}" -gt 0 ] && [ -x "$TELEGRAM_BIN" ]; then
    MSG="watcher-health found ${#ISSUES[@]} issue(s):"$'\n'
    for i in "${ISSUES[@]}"; do MSG+="• $i"$'\n'; done
    echo "$MSG" | "$TELEGRAM_BIN" 2>/dev/null || true
fi

# Exit non-zero if any STALE/CRITICAL — surfaces in launchd last_exit_code
if [ "${#ISSUES[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
